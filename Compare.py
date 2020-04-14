"""
There is an example of using OVO_ECOC class to validate on dermatology data
"""
import numpy as np
from read import read_UCI_Dataset
from read import read_Microarray_Dataset
from Classifier import OVA_ECOC,OVO_ECOC, ECOC_ONE,Dense_random_ECOC,Sparse_random_ECOC,D_ECOC,AGG_ECOC,CL_ECOC
from ECOCClassfier import ECOCClassifier2
import DataLoader
from sklearn.svm import SVC
import sklearn.metrics as ms

# ensure the result can recurrence
np.random.seed(100)
# the path of data file and you can change it
path = r'.\data_uci\Absenteeism_at_work_train.data'
# read data file
# data and label must be numpy array
data, label = read_Microarray_Dataset(path)

##########################################################################
#DataLoader
##########################################################################
dataset = 'Amazon_initial'
trainfile = "./data/" + str(dataset) + "_train.data"
testfile = "./data/" + str(dataset) + "_test.data"
validatefile = "./data/" + str(dataset) + "_validation.data"
# 其中x为特征空间，y为样本的标签
train_x, train_y, validate_x, validate_y,instance_size = DataLoader.loadDataset(trainfile,validatefile)
train_x, train_y, test_x, test_y, instance_size = DataLoader.loadDataset(trainfile, testfile)
class_size = len(np.unique(np.array(train_y)))
feature_size = len(train_x[0])
num_classifier = class_size+1

'''
,OVO_ECOC(),Sparse_random_ECOC(),CL_ECOC()
'''
__all__ = [OVA_ECOC(),Dense_random_ECOC(),D_ECOC(),AGG_ECOC()]
# initialize a OVO_ECOC object
# 'OVA_ECOC', 'OVO_ECOC', 'Dense_random_ECOC', 'Sparse_random_ECOC', 'D_ECOC', 'AGG_ECOC', 'CL_ECOC'
for item in __all__:

    E = item

    # matrix = E.create_matrix(data,label)
    matrix = E.create_matrix(train_x,train_y)
    matrix = matrix[0]

    fs_matrix = []
    for i in range(num_classifier):
        fs_matrix.append(range(feature_size))

    estimator = SVC()
    ecoc_classifier = ECOCClassifier2(estimator, matrix.tolist(), fs_matrix)
    predict_y = ecoc_classifier.fit_predict(train_x, train_y, validate_x)
    # accuracy = ms.f1_score(self.test_y,predict_y,average="micro")
    accuracy = ms.accuracy_score(validate_y, predict_y)


    # print accuracy
    print('accuracy:', accuracy)
    print('===========================================================')
