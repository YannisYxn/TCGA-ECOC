# TCGA-ECOC
Ternary Bitwise Calculator Based Genetic Algorithm for Improving Error Correcting Output Codes  
This is the implementation for paper: [A Novel Multi-Objective Genetic Algorithm Based Error Correcting Output Codes]

## Acknowledgement
* The classifier and the evaluation function is modified from [scikit-learn 0.22](https://scikit-learn.org/stable/)

## Environment
* **Windows 10 64 bit**
* **Python 3**
* [Anaconda](https://www.anaconda.com/) is strongly recommended, all necessary python packages for this project are included in the Anaconda.

## Function
The several functions of TCGA-ECOC are listed as follow:  
* generateTop: Generate first ternary operator generation of GA and the initial metric pool of ECOC metric and feature selection metric.
* generateSubCodes: According to the ternary operator metric, this function generate ECOC metric from metrics pool.
* generateSubFsMetric: According to the ternary operator metric, this function generate feature selection metric from metrics pool.
* calValue: The evaluation function of generated ECOC metric.
* cross: The cross step of GA.
* mutation: The mutation step of GA.
* calClassificationReport: Used to calculate insight to result.

## Dataset
* Data format  
Data sets of uci and microarray are both included into the folder `($root_path)/data`.
All the data sets used in experiments are splited into three parts: `xx_train.data`、`xx_validation.data` and `xx_test.data`. And the proportion of these three parts is set to 2:1:1.
These datasets are formed by setting each row as a feature and each column as an instance.
The dataset included in the folder must have no null/nan/? values.
The datasets will be loaded into algorithm by `DataLoader.py`.
* Data processing  
Feature Selection will be done by function and cleaning and scale will be done automatically.

## Runner Setup
The `TCGA-ECOC-Classifier-microarray.py` and `TCGA-ECOC-Classifier-uci.py` calls the TCGA-ECOC.  
  
The are some variables to control algorithm.
* `estimator`: base learners
* `datasets`: input datasets
* `pop_size`: size of a generation
* `pc`: possibility of cross in GA
* `pm`: possibility of mutation in GA
* `iteration`: termination number of generation in GA
* `code_pool_size`: the size of ECOC metric pool and feature selection pool

The results are collected and written in the filefolder `uci` and `microarray`.