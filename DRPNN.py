import os
import math
import numpy as np
import time
import tensorflow as tf

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DRPNN:
    def __init__(self, X_units, S_units,
                 batch_size=100,
                 hidden_units=112,
                 learning_rate_first=5.8e-05,
                 learning_rate_second=5.5e-05,
                 l1_regularization_strength=3.0e+00,
                 l2_regularization_strength=2.6e-01,
                 alpha=0.5,
                 checkpoint_path="checkpoint/DRPNN.ckpt",
                 gpu_use=False, gpu_list="0", gpu_memory_fraction=0.95,
                 random_state=None):
        self.X_units = X_units
        self.S_units = S_units
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.learning_rate_first = learning_rate_first
        self.learning_rate_second = learning_rate_second
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength
        self.alpha = alpha
        self.checkpoint_path=checkpoint_path
        self.gpu_use = gpu_use
        self.gpu_list = gpu_list
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # Set random seed
        if random_state:
            tf.set_random_seed(random_state)
    
    def fit(self, X_train, I_train, S_train, Y_train,
            X_valid, I_valid, S_valid, Y_valid,
            training_steps,
            evaluation_steps=None,
            epsilon=1e-5, threshold=0.5,
            earlystop_use=True, patience=20, earlystop_free_step=400,
            checkpoint_step=100,
            display_use=True, display_step=100):
            
        # tf.Session
        self._open_session()
        
        # tf.placeholder
        X_PH_train = tf.placeholder(shape=(None,self.X_units), dtype=tf.float32, name='X_Placeholder_train')
        I_PH_train = tf.placeholder(shape=(None,), dtype=tf.int64, name='I_Placeholder_train')
        S_PH_train = tf.placeholder(shape=(None,self.S_units), dtype=tf.float32, name='S_Placeholder_train')
        Y_PH_train = tf.placeholder(shape=(None,1), dtype=tf.uint8, name='Y_Placeholder_train')
        X_PH_valid = tf.placeholder(shape=(None,self.X_units), dtype=tf.float32, name='X_Placeholder_train')
        I_PH_valid = tf.placeholder(shape=(None,), dtype=tf.int64, name='I_Placeholder_train')
        S_PH_valid = tf.placeholder(shape=(None,self.S_units), dtype=tf.float32, name='S_Placeholder_train')
        Y_PH_valid = tf.placeholder(shape=(None,1), dtype=tf.uint8, name='Y_Placeholder_train')
        
        # tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices((X_PH_train, I_PH_train, S_PH_train, Y_PH_train))
        dataset_train = dataset_train.repeat().shuffle(buffer_size=(X_train.shape[0] + 2 * self.batch_size), seed=2018).batch(self.batch_size)
        dataset_train.prefetch(2 * self.batch_size)
        dataset_valid = tf.data.Dataset.from_tensor_slices((X_PH_valid, I_PH_valid, S_PH_valid, Y_PH_valid))
        dataset_valid = dataset_valid.repeat().batch(self.batch_size)
        dataset_valid.prefetch(2 * self.batch_size)
        
        # tf.iterator
        iterator_train = dataset_train.make_initializable_iterator()
        iterator_valid = dataset_valid.make_initializable_iterator()
        _ = self.sess.run(iterator_train.initializer,
                          feed_dict={X_PH_train:X_train,
                                     I_PH_train:I_train,
                                     S_PH_train:S_train,
                                     Y_PH_train:Y_train})
        _ = self.sess.run(iterator_valid.initializer,
                          feed_dict={X_PH_valid:X_valid,
                                     I_PH_valid:I_valid,
                                     S_PH_valid:S_valid,
                                     Y_PH_valid:Y_valid})
                                     
        # tf.iterator.handle
        handle_train = self.sess.run(iterator_train.string_handle())
        handle_valid = self.sess.run(iterator_valid.string_handle())
        
        # tf.graph
        self._output_types = dataset_train.output_types
        self._output_shapes = dataset_train.output_shapes
        self._create_graph(dataset_train.output_types, dataset_train.output_shapes)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        
        # history
        history = {
            'train':{
                'loss':[(0, np.Inf)]
            },
            'valid':{
                'loss':[(0, np.Inf)],
                'acc':[],
                'auc':[]
            }
        }
        # compute the default value of evaluation steps
        if evaluation_steps == None:
            evaluation_steps = math.ceil(X_valid.shape[0] / self.batch_size)
            
        # initialization for early stopping
        if earlystop_use:
            best_loss = np.Inf
            best_step = 0
            cnt_patience = 0
            
        # start fitting
        start_time = time.time()
        for step in range(1, training_steps+1):
            # 0) initialize terminal conditions
            termination_cond1 = False
            termination_cond2 = False
            
            # 1) training 
            _, _, loss = self.sess.run([self.TRAIN_first, self.TRAIN_second, self.LOSS],
                                        feed_dict={self.handle:handle_train, self.training:True})
            history['train']['loss'].append((step, loss))
            
            # 2) checkpoint and early stopping
            if step % checkpoint_step == 0:
                # validation loss
                mean_loss = 0.
                for _ in range(evaluation_steps):
                    loss = self.sess.run(self.LOSS, feed_dict={self.handle:handle_valid, self.training:False})
                    mean_loss += loss
                mean_loss /= evaluation_steps
                history['valid']['loss'].append((step, mean_loss))
                if display_use:
                    print("[CHECKPOINT]\tValidation loss: {:.5f}".format(history['valid']['loss'][-1][1]))                
                # early stopping
                if earlystop_use and (step > earlystop_free_step):
                    if history['valid']['loss'][-1][1] <= best_loss:
                        best_loss = history['valid']['loss'][-1][1]
                        best_step = step
                    if earlystop_use and (history['valid']['loss'][-1][1] > best_loss):
                        if cnt_patience < patience:
                            cnt_patience += 1
                        else:
                            termination_cond2 = True
                    else:
                        cnt_patience = 0
                        self.saver.save(self.sess, self.checkpoint_path)
                        if display_use:
                            print("[CHECKPOINT]\tModel saved in path: {}".format(self.checkpoint_path))
                else:
                    self.saver.save(self.sess, self.checkpoint_path)
                    if display_use:
                        print("[CHECKPOINT]\tModel saved in path: {}".format(self.checkpoint_path))

            # 3) termination
            termination_cond1 = abs(history['train']['loss'][-1][1] - history['train']['loss'][-2][1]) < epsilon
            if termination_cond1 or termination_cond2:
                # final loss
                if termination_cond1:
                    print("[STOP]\t|Loss[{:06d}]-Loss[{:06d}]|<{}".format(step,step-1,epsilon))
                    print("[STOP]\tTraining Loss[{:06d}]: {:.5f}".format(step, history['train']['loss'][-1][1]))
                    # final accuracy
                    mean_acc = 0.
                    mean_auc = 0.
                    for _ in range(evaluation_steps):
                        (acc, _), (auc, _) = self.sess.run([self.ACCURACY_second, self.AUCROC_second], feed_dict={self.handle:handle_valid, self.training:False, self.threshold:threshold})
                        mean_acc += acc
                        mean_auc += auc
                    mean_acc /= evaluation_steps
                    mean_auc /= evaluation_steps
                    history['valid']['acc'].append((step, mean_acc))
                    history['valid']['auc'].append((step, mean_auc))
                    print("[STOP]\tValidation accuracy: {:.3f}".format(mean_acc))
                    print("[STOP]\tValidation auc-roc: {:.3f}".format(mean_auc))
                    # final checkpoint
                    if not termination_cond2:
                        self.saver.save(self.sess, self.checkpoint_path)
                        print("[CHECKPOINT]\tModel saved in path: {}".format(self.checkpoint_path))
                elif termination_cond2:
                    print("[EARLYSTOP]\tLoss[{:06d}]>Loss[{:06}]".format(history['valid']['loss'][-1][0],best_step))
                    print("[EARLYSTOP]\tValidation Loss[{:06d}]: {:.5f}".format(best_step,best_loss))
                break
                
            # 4) display
            if display_use and step % display_step == 0:
                # training loss
                mean_loss = np.mean([l for (s,l) in history['train']['loss'][-display_step:]])
                # validation accuracy
                mean_acc = 0.
                mean_auc = 0.
                for _ in range(evaluation_steps):
                    (acc, _), (auc, _) = self.sess.run([self.ACCURACY_second, self.AUCROC_second], feed_dict={self.handle:handle_valid, self.training:False, self.threshold:threshold})
                    mean_acc += acc
                    mean_auc += auc
                mean_acc /= evaluation_steps
                mean_auc /= evaluation_steps
                history['valid']['acc'].append((step, mean_acc))
                history['valid']['auc'].append((step, mean_auc))
                print("[{:05d}/{:05d}]\tLoss[train]={:.5f}\tAccuracy[valid]={:.3f}\tAUC[valid]={:.3f}\t({:.3f} sec)".format(step,training_steps,mean_loss,mean_acc,mean_auc,time.time()-start_time))
                start_time = time.time()
                
        # tf.Session
        self._close_session()
        # history
        history['train']['loss'] = history['train']['loss'][1:]
        history['valid']['loss'] = history['valid']['loss'][1:]
        return history
    
    def predict(self, X_test, S_test, threshold=0.5):
        # dummy input data
        I_test = np.zeros((X_test.shape[0],), dtype=np.int32) + 999
        Y_test = np.zeros((X_test.shape[0],1), dtype=np.uint8) + 999
        # tf.Session
        self._open_session()
        # tf.placeholder
        X_PH_test = tf.placeholder(shape=(None,self.X_units), dtype=tf.float32, name='X_Placeholder_test')
        I_PH_test = tf.placeholder(shape=(None,), dtype=tf.int64, name='I_Placeholder_test')
        S_PH_test = tf.placeholder(shape=(None,self.S_units), dtype=tf.float32, name='S_Placeholder_test')
        Y_PH_test = tf.placeholder(shape=(None,1), dtype=tf.uint8, name='Y_Placeholder_test')
        # tf.dataset
        dataset_test = tf.data.Dataset.from_tensor_slices((X_PH_test, I_PH_test, S_PH_test, Y_PH_test))
        dataset_test = dataset_test.batch(self.batch_size)
        dataset_test.prefetch(2 * self.batch_size)
        # tf.iterator
        iterator_test = dataset_test.make_initializable_iterator()
        _ = self.sess.run(iterator_test.initializer,
                          feed_dict={X_PH_test:X_test,
                                     I_PH_test:I_test,
                                     S_PH_test:S_test,
                                     Y_PH_test:Y_test})
        # tf.iterator.handle
        handle_test = self.sess.run(iterator_test.string_handle())
        # tf.graph
        self._create_graph(dataset_test.output_types, dataset_test.output_shapes)
        self.saver.restore(self.sess, self.checkpoint_path)
        print("[CHECKPOINT]\tModel restored")
        # initialization of outputs
        outputs = []
        # predict per batch
        while True:
            try:
                predictions = self.sess.run(self.Y_second, feed_dict={self.handle:handle_test, self.training:False, self.threshold:threshold})
                outputs += predictions.tolist()
            except tf.errors.OutOfRangeError:
                break
        # tf.Session
        self._close_session()
        return np.array(outputs, dtype=np.uint8)
    
    def predict_proba(self, X_test, S_test):
        # dummy input data
        I_test = np.zeros((X_test.shape[0],), dtype=np.int32) + 999
        Y_test = np.zeros((X_test.shape[0],1), dtype=np.uint8) + 999
        # tf.Session
        self._open_session()
        # tf.placeholder
        X_PH_test = tf.placeholder(shape=(None,self.X_units), dtype=tf.float32, name='X_Placeholder_test')
        I_PH_test = tf.placeholder(shape=(None,), dtype=tf.int64, name='I_Placeholder_test')
        S_PH_test = tf.placeholder(shape=(None,self.S_units), dtype=tf.float32, name='S_Placeholder_test')
        Y_PH_test = tf.placeholder(shape=(None,1), dtype=tf.uint8, name='Y_Placeholder_test')
        # tf.dataset
        dataset_test = tf.data.Dataset.from_tensor_slices((X_PH_test, I_PH_test, S_PH_test, Y_PH_test))
        dataset_test = dataset_test.batch(self.batch_size)
        dataset_test.prefetch(2 * self.batch_size)
        # tf.iterator
        iterator_test = dataset_test.make_initializable_iterator()
        _ = self.sess.run(iterator_test.initializer,
                          feed_dict={X_PH_test:X_test,
                                     I_PH_test:I_test,
                                     S_PH_test:S_test,
                                     Y_PH_test:Y_test})
        # tf.iterator.handle
        handle_test = self.sess.run(iterator_test.string_handle())
        # tf.graph
        self._create_graph(dataset_test.output_types, dataset_test.output_shapes)
        self.saver.restore(self.sess, self.checkpoint_path)
        print("[CHECKPOINT]\tModel restored")
        # initialization of outputs
        outputs = []
        # predict per batch
        while True:
            try:
                probabilities = self.sess.run(self.P_second, feed_dict={self.handle:handle_test, self.training:False})
                outputs += probabilities.tolist()
            except tf.errors.OutOfRangeError:
                break
        # tf.Session
        self._close_session()
        return np.array(outputs, dtype=np.float32)
    
    def get_kernels(self, hidden_names=None):
        '''
        dense0
        dense1
        dense2
        output
        '''
        if hidden_names == None:
            hidden_names = ['dense0', 'dense1', 'dense2', 'output']
            
        # tf.Session
        self._open_session()
        # tf.graph
        self._create_graph(self._output_types, self._output_shapes)
        self.saver.restore(self.sess, self.checkpoint_path)
        print("[CHECKPOINT]\tModel restored")
        # weights
        kernel_dict = {}
        for hidden_name in hidden_names:
            weights = tf.get_default_graph().get_tensor_by_name('{}/kernel:0'.format(hidden_name))
            kernel_dict[hidden_name] = self.sess.run(weights)
        # tf.Session
        self._close_session()
        return kernel_dict
        
    def _create_graph(self, output_types, output_shapes):
        # tf.Placeholder
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training = tf.placeholder(dtype=bool, name='training_placeholder')
        self.threshold = tf.placeholder(dtype=tf.float32, name='threshold_placeholder')
        # tf.Iterator
        iterator = tf.data.Iterator.from_string_handle(string_handle=self.handle,
                                                       output_types=output_types,
                                                       output_shapes=output_shapes)
        X_batch, I_batch, S_batch, Y_batch = iterator.get_next()

        # Activation function
        nonlinear = tf.nn.sigmoid
        
        # Model
        dense0    = tf.layers.dense(inputs=X_batch, units=self.S_units, activation='linear', name='dense0')
        activate0 = nonlinear(dense0, name='activate0')
        dense1    = tf.layers.dense(inputs=tf.multiply(S_batch, activate0), units=self.hidden_units, activation='linear', name='dense1')
        bn1       = tf.layers.batch_normalization(dense1, training=self.training, name='batchnormalization1')
        activate1 = nonlinear(bn1, name='activate1')
        dense2    = tf.layers.dense(inputs=activate1, units=self.hidden_units, activation='linear', name='dense2')
        bn2       = tf.layers.batch_normalization(dense2, training=self.training, name='batchnormalization2')
        activate2 = nonlinear(bn2, name='activate2')
        
        # Output
        O_first  = tf.expand_dims(tf.reduce_sum(tf.multiply(dense0, tf.one_hot(I_batch, depth=self.S_units)), 1), axis=-1)
        O_second = tf.layers.dense(inputs=activate2, units=1, activation='linear', name='output')
        
        # LOSS
        LOSS_first = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=O_first, labels=tf.cast(Y_batch, dtype=tf.float32)))
        LOSS_second = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=O_second, labels=tf.cast(Y_batch, dtype=tf.float32)))
        self.LOSS = self.alpha*LOSS_first + (1. - self.alpha)*LOSS_second
        
        # OPTIMIZER 1st
        OPT_first = tf.train.FtrlOptimizer(learning_rate=self.learning_rate_first,
                                           l1_regularization_strength=self.l1_regularization_strength,
                                           l2_regularization_strength=self.l2_regularization_strength)
        var_list_first = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense0')
        # OPTIMIZER 2nd
        OPT_second = tf.train.AdamOptimizer(learning_rate=self.learning_rate_second)
        var_list_second = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense1')
        var_list_second += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense2')
        var_list_second += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'output')
        var_list_second += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'batchnormalization1')
        var_list_second += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'batchnormalization2')
        # MINIMIZATION
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.TRAIN_first = OPT_first.minimize(self.LOSS, var_list=var_list_first)
            self.TRAIN_second = OPT_second.minimize(self.LOSS, var_list=var_list_second)
        
        # PREDICTION
        self.P_second = tf.sigmoid(O_second)
        self.Y_second = tf.cast(self.P_second > self.threshold, tf.uint8)
        # ACCURACY
        self.ACCURACY_second = tf.metrics.accuracy(labels=Y_batch, predictions=self.Y_second)
        # AUC-ROC
        self.AUCROC_second = tf.metrics.auc(labels=Y_batch, predictions=self.P_second, curve='ROC')
        
        # SAVE AND RESTORE
        self.saver = tf.train.Saver()
        
    def _open_session(self):
        # Create a new session
        tf.reset_default_graph()
        if self.gpu_use:
            # GPU
            gpu_options = tf.GPUOptions(visible_device_list=self.gpu_list, per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            # CPU
            self.sess = tf.Session()
        
    def _close_session(self):
        self.sess.close()