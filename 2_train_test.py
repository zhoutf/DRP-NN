import argparse
import os
import math
import numpy as np
import time
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from DRPNN import DRPNN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outputdir', type=str, default='output_train_test', help="default='output_train_test'")
    parser.add_argument('-g', '--gpuuse', action='store_true', help="If not given, tensorflow processes with only CPU")
    parser.add_argument('-l', '--gpulist', type=str, default="0", help="default='0'")
    parser.add_argument('-f', '--gpumemoryfraction', type=float, default=0.4, help="default=0.4")
    parser.add_argument('-d', '--trainingdataset', type=str, default='GDSC', help="default=GDSC (=> training:GDSC, test:CCLE)")
    return parser.parse_args()

def main():
    args = get_args()
    print(args)
    
    if args.trainingdataset == "GDSC":
        drugresponseFile_train = "data/drugResponse_GDSC.txt"
        expressionFile_train = "data/geneExpression_GDSC.txt"
        fingerprintFile_train = "data/fingerprint_GDSC.csv"
        drugresponseFile_test = "data/drugResponse_CCLE.txt"
        expressionFile_test = "data/geneExpression_CCLE.txt"
        fingerprintFile_test = "data/fingerprint_CCLE.csv"
        train_test_name = "GDSC_CCLE"
        # Hyperparameters
        BATCH_SIZE = 100
        HIDDEN_UNITS = 51
        LEARNING_RATE_FIRST = 6.8e-06
        LEARNING_RATE_SECOND = 2.6e-03
        L1_REGULARIZATION_STRENGTH = 1.6e-03
        L2_REGULARIZATION_STRENGTH = 2.5e-03
    elif args.trainingdataset == "CCLE":
        drugresponseFile_train = "data/drugResponse_CCLE.txt"
        expressionFile_train = "data/geneExpression_CCLE.txt"
        fingerprintFile_train = "data/fingerprint_CCLE.csv"
        drugresponseFile_test = "data/drugResponse_GDSC.txt"
        expressionFile_test = "data/geneExpression_GDSC.txt"
        fingerprintFile_test = "data/fingerprint_GDSC.csv"
        train_test_name = "CCLE_GDSC"
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
    dataset_train = DATASET(drugresponseFile_train, expressionFile_train, fingerprintFile_train)
    dataset_test = DATASET(drugresponseFile_test, expressionFile_test, fingerprintFile_test)
    common_genes = sorted(list(set(dataset_train.get_genes()).intersection(dataset_test.get_genes())))
    dataset_train.reduce_genes(common_genes)
    dataset_test.reduce_genes(common_genes)
    print('[train] N_data: {}'.format(len(dataset_train)))
    print('[train] N_drugs: {}'.format(len(dataset_train.get_drugs(unique=True))))
    print('[train] N_cells: {}'.format(len(dataset_train.get_cells(unique=True))))
    print('[train] N_genes: {}'.format(len(dataset_train.get_genes())))
    print('[train] N_sensitivity: {}'.format(np.count_nonzero(dataset_train.get_labels()==0)))
    print('[train] N_resistance: {}'.format(np.count_nonzero(dataset_train.get_labels()==1)))
    print('[train] type_of_target: {}'.format(type_of_target(dataset_train.get_labels())))
    print('[test] N_data: {}'.format(len(dataset_test)))
    print('[test] N_drugs: {}'.format(len(dataset_test.get_drugs(unique=True))))
    print('[test] N_cells: {}'.format(len(dataset_test.get_cells(unique=True))))
    print('[test] N_genes: {}'.format(len(dataset_test.get_genes())))
    print('[test] N_sensitivity: {}'.format(np.count_nonzero(dataset_test.get_labels()==0)))
    print('[test] N_resistance: {}'.format(np.count_nonzero(dataset_test.get_labels()==1)))
    ########################################################
    ## 2. Split data into training and test
    ########################################################
	idx_train, idx_valid = train_test_split(range(len(dataset_train)), test_size=0.2,
											random_state=2018, stratify=dataset_train.get_drugs())
	# Make input data
	X_train, X_valid, X_test, genenames = make_xdata(dataset_train, dataset_test, idx_train, idx_valid, print_genenames=True)
	print('X_train.shape:', X_train.shape)
	print('X_valid.shape:', X_valid.shape)
	print('X_test.shape:', X_test.shape)
	I_train, I_valid, I_test, drugnames = make_idata(dataset_train, dataset_test, idx_train, idx_valid, print_drugnames=True)
	print('I_train.shape:', I_train.shape)
	print('I_valid.shape:', I_valid.shape)
	print('I_test.shape:', I_test.shape)
	S_train, S_valid, S_test = make_sdata(dataset_train, dataset_test, idx_train, idx_valid)
	print('S_train.shape:', S_train.shape)
	print('S_valid.shape:', S_valid.shape)
	print('S_test.shape:', S_test.shape)
	Y_train, Y_valid, Y_test = make_ydata(dataset_train, dataset_test, idx_train, idx_valid)
	print('Y_train.shape:', Y_train.shape)
	print('Y_valid.shape:', Y_valid.shape)
	print('Y_test.shape:', Y_test.shape) 
	
	########################################################
	## 3. Create a model
	########################################################
	# Create a neural network
	model = DRPNN(X_units=X_train.shape[1],
				  S_units=S_train.shape[1],
				  batch_size=BATCH_SIZE,
				  hidden_units=HIDDEN_UNITS,
				  learning_rate_first=LEARNING_RATE_FIRST,
				  learning_rate_second=LEARNING_RATE_SECOND,
				  l1_regularization_strength=L1_REGULARIZATION_STRENGTH,
				  l2_regularization_strength=L2_REGULARIZATION_STRENGTH,
				  checkpoint_path="checkpoint/{}_train_test.ckpt".format(train_test_name),
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
	
	# Get the fitted kernel matrix
	kernel_dict = model.get_kernels()
	for layername, kernel in kernel_dict.items():
		kernelFilename = os.path.join(args.outputdir, '{}_kernel_{}.txt'.format(train_test_name, layername))
		if layername == "dense0":
			tmp_df = pd.DataFrame(data=kernel, index=genenames, columns=drugnames)
			tmp_df.to_csv(kernelFilename)
		else:
			np.savetxt(fname=kernelFilename, X=kernel, delimiter=',')
	
	# Find the best threshold
	probs_valid = model.predict_proba(X_valid, S_valid)
	reals_valid = Y_valid
	accuracies_valid = []
	best_threshold = 0.
	best_acc = 0.
	for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
		preds_valid = (probs_valid>threshold).astype(np.uint8)
		ACC = np.count_nonzero(reals_valid==preds_valid) / reals_valid.shape[0]
		accuracies_valid.append((threshold, ACC))
		print("ACC[valid][{:.1f}]: {:.3f}".format(threshold, ACC))
		if ACC > best_acc:
			best_acc = ACC
			best_threshold = threshold
	with open(os.path.join(args.outputdir, '{}_threshold_valid_acc.txt'.format(train_test_name)), 'w') as fout:
		fout.write('Threshold\tACC\n')
		for (threshold, acc_valid) in accuracies_valid:
			fout.write('{:.1f}\t{}\n'.format(threshold, acc_valid))
	
	# Predict labels
	# preds = model.predict(X_test, S_test)
	probs = model.predict_proba(X_test, S_test)
	reals = Y_test
	
	# Accuracy
	preds = (probs>best_threshold).astype(np.uint8)
	ACC = np.count_nonzero(reals==preds) / reals.shape[0]
	print("ACC[test][{:.1f}]: {:.3f}".format(best_threshold, ACC))
	
	# ROC curve        
	fpr, tpr, threshold = roc_curve(reals, probs)
	tpr[0] = 0.0
	tpr[-1] = 1.0
	roc_auc = auc(fpr, tpr)
	print("AUC[test]: {:.3f}".format(roc_auc))
	
	# train loss
	with open(os.path.join(args.outputdir, '{}_train_loss.txt'.format(train_test_name)), 'w') as fout:
		fout.write('STEP\tLOSS\n')
		for (step, loss_train) in history['train']['loss']:
			fout.write('{:5d}\t{}\n'.format(step,loss_train))
			
	# valid acc, auc
	with open(os.path.join(args.outputdir, '{}_valid_acc_auc.txt'.format(train_test_name)), 'w') as fout:
		fout.write('STEP\tACC\tAUC\n')
		for (step, acc_valid), (_, auc_valid) in zip(history['valid']['acc'], history['valid']['auc']):
			fout.write('{:5d}\t{}\t{}\n'.format(step,acc_valid,auc_valid))
	
	# roc curve
	with open(os.path.join(args.outputdir, '{}_roc_curve.txt'.format(train_test_name)), 'w') as fout:
		fout.write('FPR\tTPR\n')
		for fp, tp in zip(fpr, tpr):
			fout.write('{:.6f}\t{:.6f}\n'.format(fp, tp))
			
	# results    
	with open(os.path.join(args.outputdir, '{}_predictions.txt'.format(train_test_name)), 'w') as fout:
		fout.write('Cell\tDrug\tReal\tPrediction(1:Resistance,0:Sensitivity)\n')
		for cell, drug, real, prob in zip(dataset_test.get_cells(), dataset_test.get_drugs(), reals.flatten(), probs.flatten()):
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
        return self.df.values.T # row: cell line, column: gene
        
    def get_drug2fp(self):
        return self.drug2fp
        
    def reduce_genes(self, targets):
        self.ge = self.ge.filter(targets, axis=0)
    
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

def make_SSP_df(dataset_train):
    # reference drug
    drug2fp_train = dataset_train.get_drug2fp()
    drugs_ref = dataset_train.get_drugs()
    drugs_ref = np.unique(drugs_ref)
    drugs_ref = np.sort(drugs_ref, kind='mergesort')
    fingerprints_ref = [drug2fp_train[drug] for drug in drugs_ref]
    # drug2ssp
    drug2ssp_train = dataset_train.get_drug2ssp(fingerprints_ref)
    # ssp matrix
    mat = np.array([drug2ssp_train[drug] for drug in drugs_ref], dtype=np.float32)
    # ssp data frame
    df = pd.DataFrame(data=mat, index=drugs_ref, columns=drugs_ref)
    return df
    
def make_xdata(dataset_train, dataset_test, idx_train, idx_valid, print_genenames=False):
    # train
    cells_train = dataset_train.get_cells()[idx_train]
    cells_valid = dataset_train.get_cells()[idx_valid]
    x_train = np.array([dataset_train.ge[cell].values for cell in cells_train], dtype=np.float32)
    x_valid = np.array([dataset_train.ge[cell].values for cell in cells_valid], dtype=np.float32)
    # test
    cells_test = dataset_test.get_cells()
    x_test = np.array([dataset_test.ge[cell].values for cell in cells_test], dtype=np.float32)
    # standardization
    for j in range(x_train.shape[1]):
        x_test[:,j] = (x_test[:,j] - x_test[:,j].mean() + x_train[:,j].mean()) / (x_test[:,j].std() / x_train[:,j].std())
    if print_genenames:
        genes = dataset_train.get_genes()
        return x_train, x_valid, x_test, genes
    else:
        return x_train, x_valid, x_test
    
def make_idata(dataset_train, dataset_test, idx_train, idx_valid, print_drugnames=False):
    # reference drug
    drugs_ref = dataset_train.get_drugs()[idx_train]
    drugs_ref = np.unique(drugs_ref)
    drugs_ref = np.sort(drugs_ref, kind='mergesort')
    # train
    drug2index = {drug:i for i, drug in enumerate(drugs_ref)}
    drugs_train = dataset_train.get_drugs()[idx_train]
    drugs_valid = dataset_train.get_drugs()[idx_valid]
    i_train = np.array([drug2index[drug] for drug in drugs_train], dtype=np.int32)
    i_valid = np.array([drug2index[drug] for drug in drugs_valid], dtype=np.int32)
    # test (dummy data)
    i_test = (np.zeros(len(dataset_test), dtype=np.int32) + 999)
    if print_drugnames:
        return i_train, i_valid, i_test, drugs_ref
    else:
        return i_train, i_valid, i_test
    
def make_sdata(dataset_train, dataset_test, idx_train, idx_valid):
    # reference drug
    drug2fp_train = dataset_train.get_drug2fp()
    drugs_ref = dataset_train.get_drugs()[idx_train]
    drugs_ref = np.unique(drugs_ref)
    drugs_ref = np.sort(drugs_ref, kind='mergesort')
    fingerprints_ref = [drug2fp_train[drug] for drug in drugs_ref]
    # drug2ssp
    drug2ssp_train = dataset_train.get_drug2ssp(fingerprints_ref)
    drug2ssp_test = dataset_test.get_drug2ssp(fingerprints_ref)
    # train
    drugs_train = dataset_train.get_drugs()[idx_train]
    drugs_valid = dataset_train.get_drugs()[idx_valid]
    # test
    drugs_test = dataset_test.get_drugs()
    # basis
    s_train = np.array([drug2ssp_train[drug] for drug in drugs_train], dtype=np.float32)
    s_valid = np.array([drug2ssp_train[drug] for drug in drugs_valid], dtype=np.float32)
    s_test = np.array([drug2ssp_test[drug] for drug in drugs_test], dtype=np.float32)
    return s_train, s_valid, s_test

def make_ydata(dataset_train, dataset_test, idx_train, idx_valid):
    labels_train = dataset_train.get_labels()[idx_train]
    labels_valid = dataset_train.get_labels()[idx_valid]
    labels_test = dataset_test.get_labels()
    y_train = np.expand_dims(np.array(labels_train, dtype=np.uint8), axis=-1)
    y_valid = np.expand_dims(np.array(labels_valid, dtype=np.uint8), axis=-1)
    y_test = np.expand_dims(np.array(labels_test, dtype=np.uint8), axis=-1)
    return y_train, y_valid, y_test
    
if __name__=="__main__":
    main()