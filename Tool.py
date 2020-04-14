import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''读入数据并将数据处理为所需形式'''
def ReadData():
    data = pd.read_table("./data/Amazon_initial.data",delimiter=",",header=None)
    data.fillna(method='ffill', axis=1, inplace=True)
    data = np.array(data).T
    y = data[-1]
    data = np.delete(data,-1,0)
    data = np.insert(data,0,values=y,axis=0)
    data = data.T

    X_train, X_test = train_test_split(data, test_size=0.25, random_state=42)
    X_train, X_validation = train_test_split(X_train, test_size=0.666667, random_state=42)

    train_file = open("./data/Amazon_initial_train.data","w+")
    validation_file = open("./data/Amazon_initial_validation.data","w+")
    test_file = open("./data/Amazon_initial_test.data","w+")

    for item in X_train.T:
        train_file.write(",".join(str(x) for x in item ))
        train_file.write("\n")
    for item in X_validation.T:
        validation_file.write(",".join(str(x) for x in item ))
        validation_file.write("\n")
    for item in X_test.T:
        test_file.write(",".join(str(x) for x in item ))
        test_file.write("\n")

####################################################################################

def Shuffle(dataset):
    trainfile = "./data/" + dataset + "_train.data"
    testfile = "./data/" + dataset + "_test.data"
    validatefile = "./data/" + dataset + "_validation.data"
    # 每次读入数据时打乱重新读入
    train_data =  pd.read_csv(trainfile, header=None, delimiter=',')
    test_data = pd.read_csv(testfile, header=None, delimiter=',')
    validate_data = pd.read_csv(validatefile, header=None, delimiter=',')

    train_data = np.transpose(train_data)
    test_data = np.transpose(test_data)
    validate_data = np.transpose(validate_data)

    data = np.vstack((train_data,test_data,validate_data))

    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train, validation = train_test_split(train, test_size=0.3333333, random_state=42)

    trainfile = open(trainfile,"w+")
    testfile = open(testfile, "w+")
    validatefile = open(validatefile, "w+")

    for item in train.T:
        trainfile.write(",".join(str(x) for x in item))
        trainfile.write("\n")
    for item in validation.T:
        validatefile.write(",".join(str(x) for x in item))
        validatefile.write("\n")
    for item in test.T:
        testfile.write(",".join(str(x) for x in item))
        testfile.write("\n")

    trainfile.close()
    testfile.close()
    validatefile.close()