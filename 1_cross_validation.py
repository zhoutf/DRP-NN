import argparse
import os
import math
import numpy as np
import time
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from DRPNN import DRPNN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outputdir', type=str, default='output_cv', help="default='output_cv'")
    parser.add_argument('-g', '--gpuuse', action='store_true', help="If not given, tensorflow processes with only CPU")
    parser.add_argument('-l', '--gpulist', type=str, default="0", help="default='0'")
    parser.add_argument('-f', '--gpumemoryfraction', type=float, default=0.4, help="default=0.4")
    parser.add_argument('-d', '--dataset', type=str, default='GDSC', help="default='GDSC'")
    return parser.parse_args()
    
def main():
    args = get_args()
    print(args)
    
    # Dataset
    if args.dataset == "GDSC":
        drugresponseFile = "data/drugResponse_GDSC.txt"
        expressionFile = "data/geneExpression_GDSC.txt"
        fingerprintFile = "data/fingerprint_GDSC.csv"
        # Hyperparameters
        BATCH_SIZE = 100
        HIDDEN_UNITS = 51
        LEARNING_RATE_FIRST = 6.8e-06
        LEARNING_RATE_SECOND = 2.6e-03
        L1_REGULARIZATION_STRENGTH = 1.6e-03
        L2_REGULARIZATION_STRENGTH = 2.5e-03
    elif args.dataset == "CCLE":
        drugresponseFile = "data/drugResponse_CCLE.txt"
        expressionFile = "data/geneExpression_CCLE.txt"
        fingerprintFile = "data/fingerprint_CCLE.csv"
        # Hyperparameters
        BATCH_SIZE = 100
        HIDDEN_UNITS = 112
        LEARNING_RATE_FIRST = 5.8e-05
        LEARNING_RATE_SECOND = 5.5e-05
        L1_REGULARIZATION_STRENGTH = 3.0e+00
        L2_REGULARIZATION_STRENGTH = 2.6e-01
    else:
        print("Please enter either GDSC or CCLE")
        exit(1)
    

    ########################################################
    ## 1. Read data
    ########################################################
    dataset = DATASET(drugresponseFile, expressionFile, fingerprintFile)
    print('N_data: {}'.format(len(dataset)))
    print('N_drugs: {}'.format(len(dataset.get_drugs(unique=True))))
    print('N_cells: {}'.format(len(dataset.get_cells(unique=True))))
    print('N_genes: {}'.format(len(dataset.get_genes())))
    print('N_sensitivity: {}'.format(np.count_nonzero(dataset.get_labels()==0)))
    print('N_resistance: {}'.format(np.count_nonzero(dataset.get_labels()==1)))
    ########################################################
    ## 2. Define outputs
    ########################################################
    reals = dataset.get_labels()
    preds = np.zeros_like(reals)
    probs = np.zeros_like(reals, dtype=np.float32)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    ########################################################
    ## 3. Split data into training and test
    ########################################################
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    for k, (idx_train, idx_test) in enumerate(kf.split(X=np.zeros(len(dataset)), y=dataset.get_labels())):
        idx_train, idx_valid = train_test_split(idx_train, test_size=0.2, random_state=2019, stratify=dataset.get_labels()[idx_train])
        # Make input data
        X_train, X_valid, X_test = dataset.make_xdata(idx_train, idx_valid, idx_test)
        print('X_train.shape:', X_train.shape)
        print('X_valid.shape:', X_valid.shape)
        print('X_test.shape:', X_test.shape)
        I_train, I_valid, I_test = dataset.make_idata(idx_train, idx_valid, idx_test)
        print('I_train.shape:', I_train.shape)
        print('I_valid.shape:', I_valid.shape)
        print('I_test.shape:', I_test.shape)
        S_train, S_valid, S_test = dataset.make_sdata(idx_train, idx_valid, idx_test)
        print('S_train.shape:', S_train.shape)
        print('S_valid.shape:', S_valid.shape)
        print('S_test.shape:', S_test.shape)
        Y_train, Y_valid, Y_test = dataset.make_ydata(idx_train, idx_valid, idx_test)
        print('Y_train.shape:', Y_train.shape)
        print('Y_valid.shape:', Y_valid.shape)
        print('Y_test.shape:', Y_test.shape) 
        
        # Create a neural network
        model = DRPNN(X_units=X_train.shape[1],
                      S_units=S_train.shape[1],
                      batch_size=BATCH_SIZE,
                      hidden_units=HIDDEN_UNITS,
                      learning_rate_first=LEARNING_RATE_FIRST,
                      learning_rate_second=LEARNING_RATE_SECOND,
                      l1_regularization_strength=L1_REGULARIZATION_STRENGTH,
                      l2_regularization_strength=L2_REGULARIZATION_STRENGTH,
                      checkpoint_path="checkpoint/{}_fold{}.ckpt".format(args.dataset, k),
                      gpu_use=args.gpuuse, gpu_list=args.gpulist, gpu_memory_fraction=args.gpumemoryfraction,
                      random_state=2019)
    
        # Fit using mini-batch
        steps_per_epoch = math.ceil(X_train.shape[0] / BATCH_SIZE)
        history = model.fit(X_train, I_train, S_train, Y_train, # training
                            X_valid, I_valid, S_valid, Y_valid, # valid
                            training_steps=(steps_per_epoch * 250),
                            earlystop_use=True, patience=20, earlystop_free_step=(steps_per_epoch*4),
                            checkpoint_step=(steps_per_epoch*2),
                            display_step=(steps_per_epoch*2))
        
        ## Predict labels
        preds_test = model.predict(X_test, S_test)
        probs_test = model.predict_proba(X_test, S_test)
    
        for idx, prediction in zip(idx_test, preds_test):
            preds[idx] = prediction[0]
        for idx, probability in zip(idx_test, probs_test):
            probs[idx] = probability[0]
        
        ## Accuracy
        acc = np.count_nonzero(Y_test.flatten() == preds_test.flatten()) / Y_test.shape[0]
        print("ACC[test]: {:.3f}".format(acc))
        
        ## ROC curve        
        fpr, tpr, threshold = roc_curve(Y_test, probs_test)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        tprs[-1][-1] = 1.0
        roc_auc = auc(mean_fpr, tprs[-1])
        aucs.append(roc_auc)
        print("ROC_AUC[test]: {:.3f}".format(roc_auc))
        
        ## train loss
        with open(os.path.join(args.outputdir, '{}_train_loss_fold{}.txt'.format(args.dataset,k)), 'w') as fout:
            fout.write('STEP\tLOSS\n')
            for (step, loss_train) in history['train']['loss']:
                fout.write('{:5d}\t{}\n'.format(step,loss_train))
                
        ## valid acc, auc
        with open(os.path.join(args.outputdir, '{}_valid_acc_auc_fold{}.txt'.format(args.dataset,k)), 'w') as fout:
            fout.write('STEP\tACC\tAUC\n')
            for (step, acc_valid), (_, auc_valid) in zip(history['valid']['acc'], history['valid']['auc']):
                fout.write('{:5d}\t{}\t{}\n'.format(step,acc_valid,auc_valid))
        
        ## roc curve
        with open(os.path.join(args.outputdir, '{}_roc_curve_fold{}.txt'.format(args.dataset,k)), 'w') as fout:
            fout.write('FPR\tTPR\n')
            for fp, tp in zip(mean_fpr, tprs[-1]):
                fout.write('{:.6f}\t{:.6f}\n'.format(fp, tp))
                
        ## results    
        with open(os.path.join(args.outputdir, '{}_predictions_fold{}.txt'.format(args.dataset,k)), 'w') as fout:
            fout.write('Cell\tDrug\tReal\tPrediction\n')
            for cell, drug, real, pred in zip(dataset.get_cells()[idx_test], dataset.get_drugs()[idx_test], Y_test.flatten(), probs_test.flatten()):
                fout.write('{}\t{}\t{:.6f}\t{:.6f}\n'.format(cell,drug,real,pred))
        print("----------------------------------------------------------------")
    
    ACC = np.count_nonzero(reals==preds) / reals.shape[0]
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    print("ACC: {:.3f}".format(ACC))
    print("AUC: {:.3f}".format(mean_auc))
    
    ## roc curve
    with open(os.path.join(args.outputdir, 'roc_curve_{}.txt'.format(args.dataset)), 'w') as fout:
        fout.write('FPR\tTPR\n')
        for fp, tp in zip(mean_fpr, mean_tpr):
            fout.write('{:.6f}\t{:.6f}\n'.format(fp, tp))
            
    ## results    
    with open(os.path.join(args.outputdir, 'predictions_{}.txt'.format(args.dataset)), 'w') as fout:
        fout.write('Cell\tDrug\tReal\tPrediction(1:Resistance,0:Sensitivity)\n')
        for cell, drug, real, prob in zip(dataset.get_cells(), dataset.get_drugs(), reals, probs):
            fout.write('{}\t{}\t{:.6f}\t{:.6f}\n'.format(cell,drug,real,prob))

    
   
class DATASET:
    def __init__(self, drfile, gefile, fpfile):
        ## 1. Read data
        # 1-1) Gene expression
        self.ge = pd.read_csv(gefile, sep="\t", index_col=0)
        # 1-2) Drug response
        self.dr = pd.read_csv(drfile, sep="\t", dtype='str')
        self.DRUGKEY = self.dr.columns[0]
        self.CELLKEY = self.dr.columns[1]
        self.LABELKEY = self.dr.columns[2]
        # 1-3) Fingerprint
        self.fp = pd.read_csv(fpfile, index_col=0).transpose()
        ## 2. Preprocessing
        # 2-1) Find targets
        target_drugs = self._find_target_drugs()
        target_cells = self._find_target_cells()
        # 2-2) Filter data
        self.ge = self.ge.filter(target_cells)
        self.dr = self.dr[self._get_target_idx(target_drugs, target_cells)]
        # 2-3) Label string to integer
        idx = self.dr[self.LABELKEY] == 'Resistance'
        self.dr[self.LABELKEY][idx] = 1  # Resistance == > 1
        self.dr[self.LABELKEY][~idx] = 0 # Sensitivity ==> 0
        self.dr[self.LABELKEY] = self.dr[self.LABELKEY].astype(np.uint8)

    def __len__(self):
        return len(self.dr)
        
    def get_drugs(self, unique=False):
        return self._get_series(self.DRUGKEY, unique)
        
    def get_cells(self, unique=False):
        return self._get_series(self.CELLKEY, unique)
        
    def get_labels(self, unique=False):
        return self._get_series(self.LABELKEY, unique)
        
    def get_genes(self):
        return self.ge.index.values
        
    def get_exprs(self):
        return self.df.values.T # per sample
        
    def get_drug2fp(self):
        return self.drug2fp
    
    def make_xdata(self, idx_train, idx_valid, idx_test):
        cells_train = self.get_cells()[idx_train]
        cells_valid = self.get_cells()[idx_valid]
        cells_test = self.get_cells()[idx_test]
        x_train = np.array([self.ge[cell].values for cell in cells_train], dtype=np.float32)
        x_valid = np.array([self.ge[cell].values for cell in cells_valid], dtype=np.float32)
        x_test = np.array([self.ge[cell].values for cell in cells_test], dtype=np.float32)
        return x_train, x_valid, x_test
    
    def make_idata(self, idx_train, idx_valid, idx_test):
        drugs_train = self.get_drugs()[idx_train]
        drugs_valid = self.get_drugs()[idx_valid]
        # basis
        drugs_ref = np.sort(np.unique(drugs_train), kind='mergesort')
        drug2index = {drug:i for i, drug in enumerate(drugs_ref)}
        # train
        i_train = np.array([drug2index[drug] for drug in drugs_train], dtype=np.int32)
        i_valid = np.array([drug2index[drug] for drug in drugs_valid], dtype=np.int32)
        # test (dummy data)
        i_test = (np.zeros(len(idx_test), dtype=np.int32) + 999)
        return i_train, i_valid, i_test
        
    def make_sdata(self, idx_train, idx_valid, idx_test):
        drugs_train = self.get_drugs()[idx_train]
        drugs_valid = self.get_drugs()[idx_valid]
        drugs_test = self.get_drugs()[idx_test]
        # basis
        drugs_ref = np.sort(np.unique(drugs_train), kind='mergesort')
        drug2ssp = self._make_drug2ssp(drugs_ref)
        s_train = np.array([drug2ssp[drug] for drug in drugs_train], dtype=np.float32)
        s_valid = np.array([drug2ssp[drug] for drug in drugs_valid], dtype=np.float32)
        s_test = np.array([drug2ssp[drug] for drug in drugs_test], dtype=np.float32)
        return s_train, s_valid, s_test
       
    def make_ydata(self, idx_train, idx_valid, idx_test):
        labels_train = self.get_labels()[idx_train]
        labels_valid = self.get_labels()[idx_valid]
        labels_test = self.get_labels()[idx_test]
        y_train = np.expand_dims(np.array(labels_train, dtype=np.uint8), axis=-1)
        y_valid = np.expand_dims(np.array(labels_valid, dtype=np.uint8), axis=-1)
        y_test = np.expand_dims(np.array(labels_test, dtype=np.uint8), axis=-1)
        return y_train, y_valid, y_test
        
    def get_drug2ssp(self, fingerprints_ref):
        drug2ssp = {}
        drug2fp = self.get_drug2fp()
        for x_drug, x_fp in drug2fp.items():
            x_bit = x_fp.values
            x_ssp = np.zeros(len(fingerprints_ref), dtype=np.float32)
            for j, y_bit in enumerate(fingerprints_ref):
                score = self._compute_Tanimoto_Similarity(x_bit, y_bit)
                x_ssp[j] = score
            drug2ssp[x_drug] = x_ssp
        return drug2ssp
        
    def _make_drug2ssp(self, drugs_ref):
        drug2ssp = {}
        drug2fp = self.get_drug2fp()
        fingerprints_ref = [drug2fp[drug].values for drug in drugs_ref]
        for x_drug, x_fp in drug2fp.items():
            x_bit = x_fp.values
            x_ssp = np.zeros(len(drugs_ref), dtype=np.float32)
            for j, y_bit in enumerate(fingerprints_ref):
                score = self._compute_Tanimoto_Similarity(x_bit, y_bit)
                x_ssp[j] = score
            drug2ssp[x_drug] = x_ssp
        return drug2ssp
        
    def _compute_Tanimoto_Similarity(self, x_bit, y_bit):
        numerator = np.count_nonzero(np.logical_and(x_bit, y_bit))
        denominator = np.count_nonzero(np.logical_or(x_bit, y_bit))
        return numerator / denominator
        
    def _get_series(self, KEY, unique):
        if unique:
            return np.sort(self.dr[KEY].unique(), kind='mergesort')
        else:
            return self.dr[KEY].values
    
    def _find_target_drugs(self):
        target_drugs = []
        self.drug2fp = {}
        for drugname in self.dr[self.DRUGKEY].unique():
            if drugname in self.fp:
                target_drugs.append(drugname)
                self.drug2fp[drugname] = self.fp[drugname].astype(np.uint8).astype(bool)
        return target_drugs
        
    def _find_target_cells(self):
        return self.ge.columns.intersection(self.dr[self.CELLKEY].unique())

    def _get_target_idx(self, target_drugs, target_cells):
        idx_drugs = self.dr[self.DRUGKEY].isin(target_drugs)
        idx_cells = self.dr[self.CELLKEY].isin(target_cells)
        return idx_drugs & idx_cells    

  
if __name__=="__main__":
    main()