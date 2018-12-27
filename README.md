DRP-NN: Drug Responce Prediction using Neural Networks

We introduce a novel Drug Response Prediction Neural Network (DRP-NN) model to predict anticancer drug response. DRP-NN shows robusty prediction for untrained drug response data and identifies biomarkers for drug sensitivity. DRP-NN requires gene expression profiles and drug molecular fingerprint data.

For more detail, please refer to Choi, Jonghwan, et al. "DRP-NN: a deep learning model for more accurate prediction of anticancer drug response." (submitted)


* Latest update: 27 December 2018

--------------------------------------------------------------------------------------------
SYSTEM REQUIERMENTS: 

	DRP-NN requires system memory larger than 24GB.
	
	If you want to use tensorflow-gpu, GPU memory of more than 4GB is required.

	
--------------------------------------------------------------------------------------------
USAGE: 

	$ python 1_cross_validation.py -d GDSC
	
	$ python 1_cross_validation.py -d CCLE
	
	$ python 2_train_test.py -d GDSC
	
	$ python 2_train_test.py -d CCLE


--------------------------------------------------------------------------------------------
NOTE:

    The option parameter '-h' shows help message.
	
	$ python 1_cross_validation.py -h
	
	$ python 2_train_test.py -h
	
	
--------------------------------------------------------------------------------------------
EXAMPLE 1 (1_cross_validation.py):

	$ python 1_cross_validation.py -d GDSC
	
	Namespace(gpuuse=False, outputdir='output_cv', trainingdataset='GDSC')
	N_data: 190036
	N_drugs: 222
	N_cells: 983
	N_genes: 17780
	N_sensitivity: 69430
	N_resistance: 120606
	-------- K fold cross validation 0/5 --------
	[01217/304250]  Loss[train]=0.44767     Accuracy[valid]=0.783   AUC[valid]=0.880        (75.935 sec)
	[CHECKPOINT]    Validation loss: 0.41705
	[CHECKPOINT]    Model saved in path: checkpoint/GDSC_fold0.ckpt
	[02434/304250]  Loss[train]=0.41224     Accuracy[valid]=0.795   AUC[valid]=0.879        (71.030 sec)
	[STOP]  |Loss[002556]-Loss[002555]|<1e-05
	[STOP]  Training Loss[002556]: 0.35525
	[STOP]  Validation accuracy: 0.785
	[STOP]  Validation auc-roc: 0.862
	[CHECKPOINT]    Model saved in path: checkpoint/GDSC_fold0.ckpt
	[CHECKPOINT]    Model restored
	[CHECKPOINT]    Model restored
	ACC[test]: 0.721
	AUROC[test]: 0.881
	-------- K fold cross validation 1/5 --------
	
	... (skip) ...
	
	-------- K fold cross validation 4/5 --------
	[01217/304250]  Loss[train]=0.44632     Accuracy[valid]=0.796   AUC[valid]=0.882        (140.061 sec)
	[STOP]  |Loss[001560]-Loss[001559]|<1e-05
	[STOP]  Training Loss[001560]: 0.38150
	[STOP]  Validation accuracy: 0.758
	[STOP]  Validation auc-roc: 0.853
	[CHECKPOINT]    Model saved in path: checkpoint/GDSC_fold4.ckpt
	[CHECKPOINT]    Model restored
	[CHECKPOINT]    Model restored
	ACC[test]: 0.671
	AUROC[test]: 0.862
	-------- Final results ------------------------
	mean_ACC: 0.720
	mean_AUC: 0.881
	Total execution time: 4576 sec


--------------------------------------------------------------------------------------------
EXAMPLE 2 (2_train_test.py):

	$ python 2_train_test.py -d GDSC
	
	Namespace(gpuuse=False, outputdir='output_train_test', trainingdataset='GDSC')
	GDSC dataset
	[train] N_data: 190036
	[train] N_drugs: 222
	[train] N_cells: 983
	[train] N_genes: 16017
	[train] N_sensitivity: 69430
	[train] N_resistance: 120606
	CCLE dataset
	[test] N_data: 5724
	[test] N_drugs: 12
	[test] N_cells: 491
	[test] N_genes: 16017
	[test] N_sensitivity: 344
	[test] N_resistance: 5380
	[01521/380250]  Loss[train]=0.44468     Accuracy[valid]=0.787   AUC[valid]=0.882        (87.761 sec)
	[CHECKPOINT]    Validation loss: 0.42159
	[CHECKPOINT]    Model saved in path: checkpoint/GDSC_CCLE_train_test.ckpt
	[03042/380250]  Loss[train]=0.41146     Accuracy[valid]=0.796   AUC[valid]=0.878        (85.043 sec)
	[04563/380250]  Loss[train]=0.40328     Accuracy[valid]=0.802   AUC[valid]=0.877        (78.060 sec)
	[CHECKPOINT]    Validation loss: 0.55499
	[CHECKPOINT]    Model saved in path: checkpoint/GDSC_CCLE_train_test.ckpt
	[06084/380250]  Loss[train]=0.39980     Accuracy[valid]=0.786   AUC[valid]=0.851        (85.230 sec)
	[07605/380250]  Loss[train]=0.39422     Accuracy[valid]=0.773   AUC[valid]=0.839        (77.279 sec)
	[CHECKPOINT]    Validation loss: 0.43685
	[CHECKPOINT]    Model saved in path: checkpoint/GDSC_CCLE_train_test.ckpt
	[09126/380250]  Loss[train]=0.39192     Accuracy[valid]=0.774   AUC[valid]=0.841        (87.585 sec)
	[10647/380250]  Loss[train]=0.39123     Accuracy[valid]=0.778   AUC[valid]=0.845        (77.185 sec)
	[CHECKPOINT]    Validation loss: 0.45661
	[12168/380250]  Loss[train]=0.38998     Accuracy[valid]=0.781   AUC[valid]=0.849        (84.355 sec)
	[13689/380250]  Loss[train]=0.38794     Accuracy[valid]=0.776   AUC[valid]=0.847        (77.326 sec)
	[CHECKPOINT]    Validation loss: 0.39146
	[CHECKPOINT]    Model saved in path: checkpoint/GDSC_CCLE_train_test.ckpt
	[15210/380250]  Loss[train]=0.38688     Accuracy[valid]=0.775   AUC[valid]=0.847        (85.321 sec)
	[16731/380250]  Loss[train]=0.38453     Accuracy[valid]=0.777   AUC[valid]=0.850        (77.357 sec)
	[CHECKPOINT]    Validation loss: 0.39642
	[18252/380250]  Loss[train]=0.38668     Accuracy[valid]=0.778   AUC[valid]=0.852        (84.324 sec)
	[19773/380250]  Loss[train]=0.38657     Accuracy[valid]=0.782   AUC[valid]=0.855        (75.998 sec)
	[CHECKPOINT]    Validation loss: 0.40457
	[21294/380250]  Loss[train]=0.38361     Accuracy[valid]=0.785   AUC[valid]=0.859        (82.059 sec)
	[STOP]  |Loss[022733]-Loss[022732]|<1e-05
	[STOP]  Training Loss[022733]: 0.44650
	[STOP]  Validation accuracy: 0.783
	[STOP]  Validation auc-roc: 0.859
	[CHECKPOINT]    Model saved in path: checkpoint/GDSC_CCLE_train_test.ckpt
	[CHECKPOINT]    Model restored
	[CHECKPOINT]    Model restored
	ACC[test]: 0.749
	AUC[test]: 0.744
	Total execution time: 1586 sec

	
--------------------------------------------------------------------------------------------