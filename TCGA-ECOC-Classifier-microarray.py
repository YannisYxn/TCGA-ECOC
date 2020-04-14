# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:18:17 2018

@author: yexiaona

microarray改为五折交叉验证
"""

import numpy as np
import random
import copy
import pandas as pd
import sklearn.metrics as ms
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from ECOCClassifier import SimpleECOCClassifier
from BaseClassifier import get_base_clf
from ECOCClassfier import ECOCClassifier2
import DataLoader
from sklearn.svm import SVC
import Tool
from Classifier import Sparse_random_ECOC, AGG_ECOC, Dense_random_ECOC, OVA_ECOC
from xlutils import copy as cp
import xlrd
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2,RFE,SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class GA_TOP(object):
    def __init__(self, class_size, feature_size, pop_size, pc, pm, iteration, code_pool_size,
                 train_x, train_y):
        self.class_size = class_size
        self.feature_size = feature_size
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.iteration = iteration
        self.code_pool_size = code_pool_size
        self.X = train_x
        self.Y = train_y

    '''generate first one of two code_matrix[numpy] and fs_matrix[list]'''

    def generateCode(self):
        code_matrix = []

        for i in range(self.class_size):
            temp = []
            for j in range(self.num_classifier):
                temp.append(random.randint(-1, 1))
            code_matrix.append(temp)
        code_matrix = np.array(code_matrix)

        self.legalityExamination(code_matrix)
        return code_matrix

    '''generate feature selection list'''

    def generateFS(self):
        # 一共引进5种特征选择方法
        index = random.randint(0, 4)
        # Filter方差选择法，参数threshold为方差的阈值
        if index == 0:
            indices = self.method
        # 随机
        elif index == 1:
            indices = []
            for i in range(100):
                indices.append(random.randint(0, self.feature_size - 1))
            return indices
        # Wrapper 递归特征消除法
        elif index == 2:
            indices = self.method
        # Embbed 基于惩罚项的特征选择法
        elif index == 3:
            #method = self.method3
            indices = []
            for i in range(100):
                indices.append(random.randint(0, self.feature_size - 1))
            return indices
        # 基于树模型的特征选择发
        elif index == 4:
            indices = self.method

        return indices


    '''generate first ternary operation line[list]'''

    def generateTOP(self, num_classifier):
        top_lines = []
        for i in range(self.pop_size):
            top_line = []
            for j in range(num_classifier):
                # 1-6代表5种三进制运算:加减乘(除)与或,新增一种异或翻转计算
                temp_column = []
                # 第一个矩阵的第几列
                temp_column.append(random.randint(0, self.code_pool_size - 1))
                temp_column.append(random.randint(0, self.num_classifier - 1))
                # 第二个矩阵的第几列
                temp_column.append(random.randint(0, self.code_pool_size - 1))
                temp_column.append(random.randint(0, self.num_classifier - 1))
                # 特征选择序列
                temp_column.append(self.generateFS())
                # 操作符，1-5为5种三进制运算：加减乘(除)与或，6为置零，7/8为取一半，9/10为取奇偶，11为异或翻转op
                temp_column.append(random.randint(1, 11))
                top_line.append(temp_column)
            top_lines.append(top_line)
        return top_lines

    '''generate sub_code_matrix[list[numpy]] with code_matrix operating through top'''

    def generateSubCodes(self, code_matrixs, top_lines):
        new_code_matrixs = []
        new_fs_matrixs = []
        for top_line in top_lines:
            new_code_matrix = []
            new_fs_matrix = []
            for i in range(len(top_line)):
                temp_code = []
                # 三进制运算 若非对称三进制与或计算出的结果为0，则用fscore更高的column置换
                if top_line[i][-1] in range(1, 6):
                    c1 = np.transpose([code_matrixs[top_line[i][0]][:,top_line[i][1]].tolist()])
                    c2 = np.transpose([code_matrixs[top_line[i][2]][:,top_line[i][3]].tolist()])
                    for j in range(self.class_size):
                        temp_code.append(self.topCalculate(code_matrixs[top_line[i][0]][j][top_line[i][1]],
                                                           code_matrixs[top_line[i][2]][j][top_line[i][3]],
                                                           top_line[i][-1]))
                # 全置为0，null
                elif top_line[i][-1] == 6:
                    for j in range(self.class_size):
                        temp_code.append(0)
                # 第一个矩阵某列前半部分 + 第二个矩阵某列后半部分
                elif top_line[i][-1] == 7:
                    for j in range(round(self.class_size / 2)):
                        temp_code.append(code_matrixs[top_line[i][0]][j][top_line[i][1]])
                    for j in range(round(self.class_size / 2), self.class_size):
                        temp_code.append(code_matrixs[top_line[i][2]][j][top_line[i][3]])
                # 第一个矩阵某列后半部分 + 第二个矩阵某列前半部分
                elif top_line[i][-1] == 8:
                    for j in range(round(self.class_size / 2)):
                        temp_code.append(code_matrixs[top_line[i][2]][j][top_line[i][3]])
                    for j in range(round(self.class_size / 2), self.class_size):
                        temp_code.append(code_matrixs[top_line[i][0]][j][top_line[i][1]])
                # 第一个矩阵某列奇数行 + 第二个矩阵某列偶数行
                elif top_line[i][-1] == 9:
                    for j in range(self.class_size):
                        if j % 2 == 0:
                            temp_code.append(code_matrixs[top_line[i][2]][j][top_line[i][3]])
                        else:
                            temp_code.append(code_matrixs[top_line[i][0]][j][top_line[i][1]])
                # 第一个矩阵某列偶数行 + 第二个矩阵某列奇数行
                elif top_line[i][-1] == 10:
                    for j in range(self.class_size):
                        if j % 2 == 0:
                            temp_code.append(code_matrixs[top_line[i][0]][j][top_line[i][1]])
                        else:
                            temp_code.append(code_matrixs[top_line[i][2]][j][top_line[i][3]])
                # 进行异或翻转操作，即1 op -1 = 0 ; 0 op 1 = -1 ; 0 op 1 = -1
                elif top_line[i][-1] == 11:
                    for j in range(self.class_size):
                        temp_code.append(self.topCalculate(code_matrixs[top_line[i][0]][j][top_line[i][1]],
                                                           code_matrixs[top_line[i][2]][j][top_line[i][3]],
                                                           top_line[i][-1]))
                new_code_matrix.append(temp_code)
                '''若全为0，直接去掉'''
                if top_line[i][-1] == 6:
                    del new_code_matrix[-1]
                '''生成特征矩阵'''
                temp_fs = np.zeros(self.feature_size, dtype='int').tolist()
                for j in top_line[i][4]:
                    temp_fs[j] = 1
                new_fs_matrix.append(temp_fs)
            new_code_matrix = np.transpose(new_code_matrix)
            # self.legalityExamination(new_code_matrix,self.fs_matrix)
            new_code_matrix = self.greedyExamination(new_code_matrix)
            new_code_matrixs.append(new_code_matrix)
            new_fs_matrixs.append(new_fs_matrix)
        return new_code_matrixs,new_fs_matrixs

    '''seven types tenary operating rules'''

    def topCalculate(self, a, b, operation):
        # 三进制对称加法
        if operation == 1:
            if a == -1:
                if b == -1 or b == 0:
                    return -1
                elif b == 1:
                    return 0
            elif a == 0:
                return b
            elif a == 1:
                if b == -1:
                    return 0
                elif b == 0 or b == 1:
                    return 1
        # 三进制对称减法
        elif operation == 2:
            if a == -1:
                if b == -1:
                    return 0
                elif b == 0 or b == 1:
                    return -1
            elif a == 0:
                return -b
            elif a == 1:
                if b == -1 or b == 0:
                    return 1
                elif b == 1:
                    return 0
        # 三进制对称乘法
        elif operation == 3:
            if a == -1:
                return -b
            elif a == 0:
                return 0
            elif a == 1:
                return b
        # 三进制不对称与 若结果为0，则将0更换为fscore更高的a
        elif operation == 4:
            if a == -1:
                return b
            elif a == 0:
                if b == -1 or b == 0:
                    # return 0
                    return a
                else:
                    return 1
            elif a == 1:
                return 1
        # 三进制不对称或 若结果为0，则将0更换为fscore更高的a
        elif operation == 5:
            if a == -1:
                return -1
            elif a == 0:
                if b == -1:
                    return -1
                else:
                    # return 0
                    return a
            elif a == 1:
                return b
        # 异或翻转operator
        elif operation == 11:
            if a == b:
                return a
            else:
                if a == 0:
                    if b == 1:
                        return -1
                    elif b == -1:
                        return 1
                elif a == 1:
                    if b == 0:
                        return -1
                    elif b == -1:
                        return 0
                elif a == -1:
                    if b == 0:
                        return 1
                    elif b == 1:
                        return 0


    '''calculate the scores of current code_matrix'''

    def calValue(self, code_matrix, fs_matrix, dataType):
        # estimator = KNeighborsClassifier(n_neighbors=3)
        # estimator = DecisionTreeClassifier()
        estimator = SVC()

        ecoc_classifier = ECOCClassifier2(estimator, code_matrix.tolist(), fs_matrix)
        if dataType == "validate":
            self.train_x, self.validate_x, self.train_y, self.validate_y = train_test_split(self.X, self.Y, test_size=0.2)
            predict_y = ecoc_classifier.fit_predict(self.train_x, self.train_y, self.validate_x)
            accuracy = ms.f1_score(self.validate_y, predict_y, average="micro")
            # accuracy = ms.accuracy_score(self.validate_y, predict_y)
        elif dataType == "test":
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y, test_size=0.2)
            predict_y = ecoc_classifier.fit_predict(self.train_x, self.train_y, self.test_x)
            accuracy = ms.f1_score(self.test_y, predict_y, average="micro")
            # accuracy = ms.accuracy_score(self.test_y, predict_y)
        return accuracy

    def calClassificationReport(self, code_matrix, fs_matrix, dataType):
        # estimator = KNeighborsClassifier(n_neighbors=3)
        # estimator = DecisionTreeClassifier()
        estimator = SVC()

        ecoc_classifier = ECOCClassifier2(estimator, code_matrix.tolist(), fs_matrix)
        if dataType == "validate":
            self.train_x, self.validate_x, self.train_y, self.validate_y = train_test_split(self.X, self.Y, test_size=0.2)
            predict_y = ecoc_classifier.fit_predict(self.train_x, self.train_y, self.validate_x)
            report = classification_report(y_true=self.validate_y,y_pred=predict_y,target_names=np.unique(self.validate_y).tolist())
        elif dataType == "test":
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y, test_size=0.2)
            predict_y = ecoc_classifier.fit_predict(self.train_x, self.train_y, self.test_x)
            report = classification_report(y_true=self.test_y, y_pred=predict_y,
                                           target_names=np.unique(self.test_y).tolist())
        return report

    '''计算效果最好的矩阵类之间的距离之和'''

    def calDistance(self, code_matrix):
        distances = 0
        for i in range(self.class_size):
            distance = 0
            for j in range(i + 1, self.class_size):
                for k in range(code_matrix.shape[1]):
                    if code_matrix[i][k] != code_matrix[j][k]:
                        distance = distance + 1
            distances = distances + distance
        return distances

    '''sort'''

    def sort(self, top_lines, top_code_matrixs, top_values, top_fs_matrixs):
        for i in range(len(top_values) - 1):
            for j in range(len(top_values) - i - 1):
                if (top_values[j] < top_values[j + 1]):
                    temp = top_values[j]
                    top_values[j] = top_values[j + 1]
                    top_values[j + 1] = temp
                    temp = top_lines[j]
                    top_lines[j] = top_lines[j + 1]
                    top_lines[j + 1] = temp
                    temp = top_code_matrixs[j]
                    top_code_matrixs[j] = top_code_matrixs[j + 1]
                    top_code_matrixs[j + 1] = temp
                    temp = top_fs_matrixs[j]
                    top_fs_matrixs[j] = top_fs_matrixs[j + 1]
                    top_fs_matrixs[j + 1] = temp

    '''generate the index list of cross/mutation'''

    def generateIndex(self, p, num_classifier):
        Count = round(num_classifier * p)
        if Count == 0:
            Count = 1
        counter = 0
        index = []
        while (counter < Count):
            tempInt = random.randint(0, num_classifier - 1)
            if tempInt not in index:
                index.append(tempInt)
                counter = counter + 1
        return index

    '''cross over'''

    def cross(self, top_lines, num_classifier):
        temp_lines = copy.deepcopy(top_lines)
        index_individual = self.generateIndex(self.pc, self.pop_size)
        for i in index_individual:
            index_operator = self.generateIndex(self.pc, num_classifier)
            for j in index_operator:
                # 以operator为单位交换individual
                temp = temp_lines[i - 1][j]
                temp_lines[i - 1][j] = temp_lines[i][j]
                temp_lines[i][j] = temp
                if len(temp_lines[i-1][j][4])<len(temp_lines[i][j][4]):
                    size1 = len(temp_lines[i-1][j][4])
                    size2 = len(temp_lines[i][j][4])
                    index_fs = self.generateIndex(0.2, size1)
                    for k in index_fs:
                        temp_index = random.randint(0, size2 - 1)
                        temp_fs = temp_lines[i - 1][j][4][k]
                        temp_lines[i - 1][j][4][k] = temp_lines[i][j][4][temp_index]
                        temp_lines[i][j][4][temp_index] = temp_fs
                else:
                    size1 = len(temp_lines[i][j][4])
                    size2 = len(temp_lines[i-1][j][4])
                    index_fs = self.generateIndex(0.2, size1)
                    for k in index_fs:
                        temp_index = random.randint(0, size2 - 1)
                        temp_fs = temp_lines[i][j][4][k]
                        temp_lines[i][j][4][k] = temp_lines[i - 1][j][4][temp_index]
                        temp_lines[i - 1][j][4][temp_index] = temp_fs
        return temp_lines

    '''mutation'''

    def mutation(self, top_lines, num_classifier):
        index_individual = self.generateIndex(self.pm, self.pop_size)
        for i in index_individual:
            index = self.generateIndex(self.pm, num_classifier)
            for j in index:
                # 突变个体
                index_in_column = self.generateIndex(random.randint(1, 4) / 6, 6)
                for k in index_in_column:
                    if k == 0 or k == 2:
                        tempInt = random.randint(0, self.code_pool_size - 1)
                        while tempInt == top_lines[i][j][k]:
                            tempInt = random.randint(0, self.code_pool_size - 1)
                    elif k == 1 or k == 3:
                        tempInt = random.randint(0, self.num_classifier - 1)
                        while tempInt == top_lines[i][j][k]:
                            tempInt = random.randint(0, self.num_classifier - 1)
                    elif k == 4:
                        index_fs = self.generateIndex(0.01, len(top_lines[i][j][k]))
                        for l in index_fs:
                            top_lines[i][j][k][l] = random.randint(0, self.feature_size - 1)
                        tempInt = top_lines[i][j][k]
                    elif k == 5:
                        tempInt = random.randint(1, 11)
                        while tempInt == top_lines[i][j][k]:
                            tempInt = random.randint(1, 11)
                    top_lines[i][j][k] = tempInt

    '''贪心合法性检查'''

    def greedyExamination(self, code_matrix):
        flag = False
        index = 0
        while index < code_matrix.shape[1]:
            # 每一列必须包含1和-1
            # 若该列全为1或全为-1
            if (code_matrix[:, index] == 1).all() == True or (code_matrix[:, index] == -1).all() == True:
                flag = True
                # 如果列超过1.5*class_size则删除该列
                if code_matrix.shape[1] >= round(self.class_size * 1.5):
                    code_matrix = np.delete(code_matrix, index, axis=1)
                    continue
                # 列未超过1.5*class_size则取前2/n个取反
                else:
                    bound = round(code_matrix.shape[0] / 2)
                    if bound == 0:
                        bound = 1
                    for j in range(bound + 1):
                        if j % 2 == 0:
                            code_matrix[j][index] = 0
                        code_matrix[j][index] = -1 * code_matrix[j][index]
            # 若全为0
            elif (code_matrix[:, index] == 0).all() == True:
                flag = True
                if code_matrix.shape[1] >= round(self.class_size * 1.5):
                    code_matrix = np.delete(code_matrix, index, axis=1)
                    continue
                else:
                    for j in range(code_matrix.shape[0]):
                        if j % 2 == 0:
                            code_matrix[j][index] = 1
                        else:
                            code_matrix[j][index] = -1
            # 全为0和-1
            elif (code_matrix[:, index] == 1).any() == False:
                flag = True
                column = code_matrix[:, index].tolist()
                position = column.index(0)
                code_matrix[position, index] = 1
            # 全为0和1
            elif (code_matrix[:, index] == -1).any() == False:
                flag = True
                column = code_matrix[:, index].tolist()
                position = column.index(0)
                code_matrix[position, index] = -1
            # 不能含有相同或相反的列
            temparray = np.zeros(self.class_size)
            transpose_code_matrix = np.transpose(code_matrix)
            temp_code_matrix = np.delete(transpose_code_matrix, index, axis=0)
            flag2 = False
            for j in range(temp_code_matrix.shape[0]):
                if ((transpose_code_matrix[index] - temp_code_matrix[j]) == temparray).all() or (
                        (transpose_code_matrix[index] + temp_code_matrix[j]) == temparray).all():
                    flag = True
                    # 如果列超出1.5*class_size则删除该列
                    if code_matrix.shape[1] >= round(self.class_size * 1.5):
                        code_matrix = np.delete(code_matrix, index, axis=1)
                        flag2 = True
                        break
                    # 如果列未超出则取前2/n个取反
                    else:
                        bound = round(code_matrix.shape[0] / 2)
                        if bound == 0:
                            bound = 1
                        for j in range(bound + 1):
                            if code_matrix[j][index] == 1:
                                code_matrix[j][index] = 0
                            elif code_matrix[j][index] == 0:
                                code_matrix[j][index] = -1
                            elif code_matrix[j][index] == -1:
                                code_matrix[j][index] = 1
            if flag2 == True:
                continue
            index = index + 1
        i = 0
        temparray = np.zeros(code_matrix.shape[1])
        for line in code_matrix:
            # 不能含有全为0的行
            if (line == temparray).all():
                flag = True
                bound = round(code_matrix.shape[1] / 2)
                #                if bound==0:
                #                    bound=1
                for j in range(bound + 1):
                    if j % 2 == 0:
                        line[j] = 1
                    else:
                        line[j] = -1
            # 不能含有相同的行
            temp_code_matrix = np.delete(code_matrix, i, axis=0)
            for j in range(temp_code_matrix.shape[0]):
                if ((line - temp_code_matrix[j]) == temparray).all():
                    flag = True
                    bound = round(code_matrix.shape[1] / 2)
                    if bound == 0 and len(line) > 1:
                        bound = 1
                    for j in range(bound + 1):
                        if line[j] == 1:
                            line[j] = 0
                        elif line[j] == 0:
                            line[j] = -1
                        elif line[j] == -1:
                            line[j] = 1
            i = i + 1
        if flag == True:
            self.greedyExamination(code_matrix)
        return code_matrix

    '''examination of legality突变合法性检查'''

    def legalityExamination(self, code_matrix):
        i = 0
        flag = False
        temparray = np.zeros(code_matrix.shape[1])
        for line in code_matrix:
            # 不能含有全为0的行
            while (line == temparray).all():
                index = random.randint(0, code_matrix.shape[1] - 1)
                line[index] = random.randint(-1, 1)
                flag = True
            # 不能含有相同的行
            temp_code_matrix = np.delete(code_matrix, i, axis=0)
            for j in range(temp_code_matrix.shape[0]):
                while ((line - temp_code_matrix[j]) == temparray).all():
                    index = random.randint(0, code_matrix.shape[1] - 1)
                    line[index] = random.randint(-1, 1)
                    flag = True
            i = i + 1

        temparray = np.zeros(self.class_size)
        for i in range(code_matrix.shape[1]):
            # 每一列必须包含1和-1
            while (code_matrix[:, i] == 1).any() == False or (code_matrix[:, i] == -1).any() == False:
                index = random.randint(0, self.class_size - 1)
                code_matrix[index][i] = random.randint(-1, 1)
                flag = True
            # 不能含有相同或相反的列
            transpose_code_matrix = np.transpose(code_matrix)
            temp_code_matrix = np.delete(transpose_code_matrix, i, axis=0)
            for j in range(temp_code_matrix.shape[0]):
                while ((transpose_code_matrix[i] - temp_code_matrix[j]) == temparray).all() or (
                        (transpose_code_matrix[i] + temp_code_matrix[j]) == temparray).all():
                    index = random.randint(0, self.class_size - 1)
                    transpose_code_matrix[i][index] = random.randint(-1, 1)
                    code_matrix[index][i] = transpose_code_matrix[i][index]
                    flag = True
        if flag == True:
            self.legalityExamination(code_matrix)

    '''main'''
    def main(self):

        code_matrixs = []

        self.method0 = VarianceThreshold(threshold=1).fit(self.X)
        self.method2 = RFE(estimator=LogisticRegression(), n_features_to_select=100).fit(self.X,self.Y)
        #self.method3 = SelectFromModel(LogisticRegression(penalty="l1",C=(100/self.feature_size))).fit(self.train_x,self.train_y)
        self.method4 = SelectFromModel(GradientBoostingClassifier()).fit(self.X,self.Y)

        temp0 = self.method0.get_support(indices=True).tolist()
        temp2 = self.method2.get_support(indices=True).tolist()
        temp4 = self.method4.get_support(indices=True).tolist()
        method = temp0 + temp2 + temp4
        self.method = np.unique(method)


        self.num_classifier = round(self.class_size * 2)
        for i in range(self.code_pool_size):
            code_matrix = self.generateCode()
            code_matrixs.append(code_matrix)



        top_lines = self.generateTOP(self.num_classifier)

        accuracy = []  # best accuracy of each ietration
        test_accuracy = []  # best result for test data
        avg_accuracy = []
        avg_test_accuracy = []
        distances = []

        '''begin GA'''
        for i in range(self.iteration):
            top_code_matrixs,top_fs_matrixs = self.generateSubCodes(code_matrixs, top_lines)
            top_values = []
            for j in range(len(top_code_matrixs)):
                top_values.append(self.calValue(top_code_matrixs[j], top_fs_matrixs[j], "validate"))
            print("top_lines对应的子矩阵分数:")
            print(top_values)

            '''sort top lines according to top_values'''
            self.sort(top_lines, top_code_matrixs, top_values, top_fs_matrixs)
            accuracy.append(max(top_values))
            w_sheet2.write(Index*2-1,i+1,max(top_values))

            avg_accuracy.append(sum(accuracy) / len(accuracy))

            print("最大值:")
            print(max(top_values))

            test_value = self.calValue(top_code_matrixs[0], top_fs_matrixs[0], "test")
            w_sheet2.write(Index * 2 , i + 1, test_value)
            best_fs_matrixs = top_fs_matrixs[0]
            best_matrix = top_code_matrixs[0]

            test_accuracy.append(test_value)
            avg_test_accuracy.append(sum(test_accuracy) / len(test_accuracy))

            distances.append(self.calDistance(top_code_matrixs[0]))

            temp_top_lines = copy.deepcopy(top_lines)
            random.shuffle(temp_top_lines)
            son_top_lines = self.cross(temp_top_lines, self.num_classifier)
            self.mutation(son_top_lines, self.num_classifier)
            son_code_matrixs,son_fs_matrixs = self.generateSubCodes(code_matrixs, son_top_lines)
            son_values = []
            for j in range(len(son_code_matrixs)):
                son_values.append(self.calValue(son_code_matrixs[j], son_fs_matrixs[j], "validate"))
            self.sort(son_top_lines, son_code_matrixs, son_values, son_fs_matrixs)
            print("新生成的子矩阵分数:")
            print(son_values)
            print(son_top_lines)
            if top_values[0] >= son_values[-1]:
                print("精英保留:")
                print(top_lines[0])
                print(top_values[0])
                son_values[-1] = top_values[0]
                son_top_lines[-1] = top_lines[0]
            top_lines = son_top_lines
            random.shuffle(top_lines)

        # 记录初始矩阵分数
        for i in range(len(code_matrixs)):
            value = self.calValue(code_matrixs[i],best_fs_matrixs,"test")
            w_sheet.write(Index, i+1, value)

        # 记录insight_to_result
        insightFile = open('./insight_to_result.txt',"a")
        print(dataset,file=insightFile)
        print(self.calClassificationReport(best_matrix,best_fs_matrixs,"test"),file=insightFile)
        insightFile.close()


        return accuracy, test_accuracy, avg_accuracy, avg_test_accuracy, distances


'''
    'ecoli','abalone','Absenteeism_at_work','Amazon_initial', 'thyroid', 'vertebral',
             'cmc'
            'dermatology','glass','wine', 'yeast', 'zoo','GCM','Lung2',
            
			"Breast", "GCM","Leukemia1",,"Cancers","DLBCL""Lung1","SRBCT",'DLBCL',
			'Breast','Cancers','Lung1','Lung2','SRBCT','Leukemia1',,'Lung2'
			
			'Lung1','Leukemia1','DLBCL',
'Leukemia2','GCM','Breast','Cancers',
			
            '''
datasets = ['Lung2','SRBCT'
            ]
rbook = xlrd.open_workbook('./figure.xls',formatting_info=False)
wbook = cp.copy(rbook)
w_sheet = wbook.get_sheet(0)
w_sheet2 = wbook.get_sheet(1)
Index = 1
for dataset in datasets:
    for i in range(1):
        Row = dataset + str(i)
        w_sheet.write(Index,0,Row)
        w_sheet2.write(Index*2-1,0,Row+'validate')
        w_sheet2.write(Index*2,0,Row+'test')
        ##########################################################################
        # DataLoader
        ##########################################################################
        trainfile = "./data_microarray/" + str(dataset) + ".data"
        # trainfile = "./data/" + str(dataset) + "_train.data"
        # testfile = "./data/" + str(dataset) + "_test.data"
        # validatefile = "./data/" + str(dataset) + "_validation.data"
        # # 每次读入数据时打乱重新读入
        # Tool.Shuffle(str(dataset))

        # 其中x为特征空间，y为样本的标签
        # train_x, train_y, validate_x, validate_y, instance_size = DataLoader.loadDataset(trainfile, validatefile)
        # train_x, train_y, test_x, test_y, instance_size = DataLoader.loadDataset(trainfile, testfile)
        train_x, train_y, instance_size = DataLoader.loadDataset_microarray(trainfile)

        ##########################################################################
        # 配置参数运行程序
        ##########################################################################
        class_size = len(np.unique(np.array(train_y)))
        feature_size = len(train_x[0])
        pop_size = 50  # 种群个体数量
        pc = 0.8
        pm = 0.1
        iteration = 100
        code_pool_size = 25
        ga_top = GA_TOP(class_size, feature_size, pop_size, pc, pm, iteration, code_pool_size,
                        train_x, train_y)

        accuracy, test_accuracy, avg_accuracy, avg_test_accuracy, distances = ga_top.main()
        '''print out the result'''
        accuracy.sort()
        avg_accuracy.sort()
        test_accuracy.sort()
        avg_test_accuracy.sort()
        plt.plot(accuracy, label="Validate")
        plt.plot(avg_accuracy,label = "Va",color = "g")
        plt.plot(test_accuracy, label="Test", color="r")
        plt.plot(avg_test_accuracy,label="Ta",color="m")
        plt.title("top accuracy")
        plt.legend()

        filename = "./figures/" + str(dataset) + str(i + 1) + ".png"
        plt.savefig(filename)
        plt.show()


        recordfilename = "./microarray/" + str(dataset) + ".txt"
        recordfile = open(recordfilename, "a")
        print(i, file=recordfile)
        print("ga_top最大值:", file=recordfile)
        print(max(accuracy), file=recordfile)
        print(max(test_accuracy), file=recordfile)
        print("ga_top平均值:", file=recordfile)
        print(sum(accuracy) / len(accuracy), file=recordfile)
        print(sum(test_accuracy) / len(test_accuracy), file=recordfile)
        recordfile.close()
        wbook.save('./figure.xls')
        Index = Index + 1