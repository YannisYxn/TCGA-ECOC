"""
@created 2018/10/26
@author yexiaona
"""
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVR

class FeatureSelection(object):

    '''去掉取值变化小的特征'''
    '''弃用！！！！！！！！！！！'''
    def varianceThreshold(self):
        allData = np.vstack((self.train,self.validation,self.test))
        # 若某特征为0或为1的值超过80%
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        allData = sel.fit_transform(allData)
        return self.train_validation_test_split(allData)

    '''卡方检验'''
    def Chi2(self):
        print(self.X.shape)
        X_new = SelectKBest(chi2,k=10).fit_transform(self.X,self.y)
        self.X = X_new
        self.y = y
        print(self.X.shape)
        y_new = []
        y_new.append(self.y)
        y_new = np.array(y_new)
        print(y_new.T.shape)
        self.allData = np.hstack((y_new.T,self.X))

    '''基于L1的特征选择'''
    def L1BasedFS(self):
        print(self.X.shape)
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(self.X,self.y)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(self.X)
        self.X = X_new
        y_new = []
        y_new.append(self.y)
        y_new = np.array(y_new)
        print(self.X.shape)
        self.allData = np.hstack((y_new.T,self.X))


    '''递归特征删除算法'''
    def recursiveFE(self):
        print(self.X.shape)
        # use linear regression as the model
        lr = LinearRegression
        estimator = SVR(kernel="linear")
        # rank all features
        selector = RFE(estimator, n_features_to_select=self.X.shape[1], step = 1)
        y = self.YDict_str2float(self)
        selector = selector.fit(X,y)
        y = self.YDict_float2str(self)
        model = SelectFromModel(selector, prefit=True)
        self.X = model.transform(self.X)
        self.X = X_new
        y_new = []
        y_new.append(self.y)
        y_new = np.array(y_new)
        print(self.X.shape)
        self.allData = np.hstack((y_new.T, self.X))

    '''按1:2:1比例分割train:validation:test'''
    def train_validation_test_split(self):
        X_train, X_test = train_test_split(self.allData, test_size=0.25)
        X_train, X_validation = train_test_split(X_train, test_size=0.33)
        return X_train.T, X_validation.T, X_test.T

    '''读入train,validation,test并合为一个数据集'''
    def mergeData(self,dataset):
        # 读入数据集
        train_file = open('./new_data/' + dataset + '_train.data')
        validation_file = open('./new_data/' + dataset + '_validation.data')
        test_file = open('./new_data/' + dataset + '_test.data')

        # 将数据集转化为numpy.array,因为有不是数值的值，不可使用np.loadtxt
        lines = train_file.readlines()
        train = []
        for line in lines:
            train.append(line[:-1].split(","))
        lines = validation_file.readlines()
        validation = []
        for line in lines:
            validation.append(line[:-1].split(","))
        lines = test_file.readlines()
        test = []
        for line in lines:
            test.append(line[:-1].split(","))
        # 类型转化为numpy.array
        train_y = train[0]
        test_y = test[0]
        validation_y = validation[0]
        train = np.array(train)[1:]
        validation = np.array(validation)[1:]
        test = np.array(test)[1:]

        # 当数据集中全为数值类而无文本类，可以直接使用np.loadtxt将数据集读入，而无需转换
        # train = np.loadtxt('./data/'+dataset+'_train.data',delimiter=",")
        # validation = np.loadtxt('./data/'+dataset+'_validation.data',delimiter=",")
        # test = np.loadtxt('./data/'+dataset+'_test.data',delimiter=",")

        # 转换为每列代表特征，每行代表一个example
        train = train.T
        validation = validation.T
        test = test.T

        X = np.vstack((train,validation,test))
        y = np.hstack((train_y,validation_y,test_y))
        self.X = X.astype(float)
        self.y = y
        return X,y

    '''将字符型y变换为数值型y'''
    def YDict_str2float(self):
        ydict = {}
        index = 0
        y_new = []
        for i in self.y:
            if i in ydict.keys():
                y_new.append(ydict[i])
            else:
                ydict[i] = index
                index = index + 1
                y_new.append(ydict[i])
        self.ydict = ydict
        y_new = np.array(y_new)
        self.y = y_new

    '''将数值型y变换为字符型y'''
    def YDict_float2str(self):
        y_new = []
        for i in self.y:
            for keys in self.ydict.keys():
                if i == self.ydict[keys]:
                    y_new.append(keys)
                    break
        y_new = np.array(y_new)
        self.y = y_new

#"Absenteeism_at_work","GCM","Leukemia1","Leukemia2","arrhythmia"

datasets = ["Amazon_initial"]
for dataset in datasets:

    print(dataset)

    '''去掉取值变化小的特征'''
    # fs = FeatureSelection(train,validation,test)
    # train,validation,test = fs.varianceThreshold()

    fs = FeatureSelection
    X,y = fs.mergeData(fs,dataset=dataset)
    print(X)
    '''卡方验证'''
    fs.Chi2(fs)
    '''基于L1的特征选择'''
    # fs.L1BasedFS(fs)
    '''递归特征删除算法'''
    # fs.recursiveFE(fs)
    train,validation,test = fs.train_validation_test_split(fs)
    print(train.shape)

    train_file = open("./data/"+dataset+"_train.data","w+")
    validation_file = open("./data/"+dataset+"_validation.data","w+")
    test_file = open("./data/"+dataset+"_test.data","w+")

    for item in train:
        train_file.write(",".join(item))
        train_file.write("\n")
    for item in validation:
        validation_file.write(",".join(item))
        validation_file.write("\n")
    for item in test:
        test_file.write(",".join(item))
        test_file.write("\n")

