DRP-NN: Drug Responce Prediction using Neural Networks

We introduce a novel Drug Response Prediction Neural Network (DRP-NN) model to predict anticancer drug response. DRP-NN shows robusty prediction for untrained drug response data and identifies biomarkers for drug sensitivity. DRP-NN requires gene expression profiles and drug molecular fingerprint data.

For more detail, please refer to Choi, Jonghwan, et al. "DRP-NN: a deep learning model for more accurate prediction of anticancer drug response." (submitted)


* Latest update: 26 December 2018

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
Note:

    The option parameter '-h' shows help message.
	
	$ python 1_cross_validation.py -h
	
	$ python 2_train_test.py -h
	
	
--------------------------------------------------------------------------------------------