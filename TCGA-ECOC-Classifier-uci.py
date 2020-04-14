# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:18:17 2018

@author: yexiaona

当该列分数小于该代个体平均分数时，突变该特征选择
uci的特征选择改为在结构体中而不是在矩阵选择池中
"""

import numpy as np
import random
import copy
import sklearn.metrics as ms
from ECOCClassfier import ECOCClassifier2
import DataLoader
from sklearn.svm import SVC
import Tool
from xlutils import copy as cp
import xlrd
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class GA_TOP(object):
    def __init__(self, class_size, feature_size, pop_size, pc, pm, iteration, code_pool_size,
                 train_x, train_y, validate_x, validate_y, test_x, test_y):
        self.class_size = class_size
        self.feature_size = feature_size
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.iteration = iteration
        self.code_pool_size = code_pool_size
        self.train_x = train_x
        self.train_y = train_y
        self.validate_x = validate_x
        self.validate_y = validate_y
        self.test_x = test_x
        self.test_y = test_y

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

    '''generate feature selection indices'''
    def generateFSIndices(self):
        indices = []
        for i in range(self.feature_size):
            indices.append(random.randint(0,1))
        if 1 not in indices:
            index = random.randint(0,self.feature_size-1)
            indices[index] = 1
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
                # 特征子集矩阵fss中的第几个特征子集，初始化为每个operator对应一个
                # temp_column.append(i * num_classifier + j)
                # 【修改：将随机特征子集改为 指定某种特征选择方法共7种】
                temp_column.append(self.generateFSIndices())
                # 操作符，1-5为5种三进制运算：加减乘(除)与或，6为置零，7/8为取一半，9/10为取奇偶，11为异或翻转op
                temp_column.append(random.randint(1, 11))
                top_line.append(temp_column)
            top_lines.append(top_line)
        return top_lines

    '''generate sub_code_matrix[list[numpy]] with code_matrix operating through top'''

    def generateSubCodes(self, code_matrixs, top_lines):
        new_code_matrixs = []
        top_fs = self.generateSubFsMatrix(top_lines)
        for top_line in top_lines:
            new_code_matrix = []
            for i in range(len(top_line)):
                temp_code = []
                # 三进制运算 若非对称三进制与或计算出的结果为0，则用fscore更高的column置换
                if top_line[i][-1] in range(1, 6):
                    c1 = np.transpose([code_matrixs[top_line[i][0]][:,top_line[i][1]].tolist()])
                    c2 = np.transpose([code_matrixs[top_line[i][2]][:,top_line[i][3]].tolist()])
                    c1score = self.calValue(code_matrix=c1, fs_matrix=top_fs[i], dataType="validate")
                    c2score = self.calValue(code_matrix=c2, fs_matrix=top_fs[i], dataType="validate")
                    # 将fscore更高的column放到a位置
                    if c1score >= c2score:
                        for j in range(self.class_size):
                            temp_code.append(self.topCalculate(code_matrixs[top_line[i][0]][j][top_line[i][1]],
                                                               code_matrixs[top_line[i][2]][j][top_line[i][3]],
                                                               top_line[i][-1]))
                    else:
                        for j in range(self.class_size):
                            temp_code.append(self.topCalculate(code_matrixs[top_line[i][2]][j][top_line[i][3]],
                                                               code_matrixs[top_line[i][0]][j][top_line[i][1]],
                                                               top_line[i][-1]))
                    # for j in range(self.class_size):
                    #     temp_code.append(self.topCalculate(code_matrixs[top_line[i][0]][j][top_line[i][1]],
                    #                                        code_matrixs[top_line[i][2]][j][top_line[i][3]],
                    #                                        top_line[i][-1]))
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
            new_code_matrix = np.transpose(new_code_matrix)
            # self.legalityExamination(new_code_matrix,self.fs_matrix)
            new_code_matrix = self.greedyExamination(new_code_matrix)
            new_code_matrixs.append(new_code_matrix)
        return new_code_matrixs

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

    '''generate sub feature selection matrix fs_matrix'''

    def generateSubFsMatrix(self, top_lines):
        # 每个top_line中的每个operator的第五个结构指向特征子集
        fs_matrixs = []
        for i in range(len(top_lines)):
            fs_matrix = []
            for j in range(len(top_lines[i])):
                fs_matrix.append(top_lines[i][j][4])
            fs_matrixs.append(fs_matrix)
        return fs_matrixs

    # def generateNewData(self, top_lines, data_x, data_y):
    #     for i in range(len(top_lines)):
    #         for j in range(len(top_lines[i])):
    #             index = top_lines[i][j][4]  # 指向某种特定的特征选择方法
    #             # 方差选择法
    #             if index == 0:
    #                 data = feature_selection.VarianceThreshold(threshold=3).fit_transform(data_x)
    #             # 相关系数法
    #             elif index == 1:
    #                 data = feature_selection.SelectKBest(feature_selection.chi2, k=0.7*len(data_x[0])).fit_transform(data_x,data_y)
    #             # 递归特征消除法
    #             elif index == 2:
    #                 data = feature_selection.RFE(estimator=LogisticRegression(),n_features_to_select=0.7*len(data_x[0])).fit_transform(data_x,data_y)
    #             # 基于惩罚项的特征选择法
    #             elif index == 3:
    #                 data = feature_selection.SelectFromModel(LogisticRegression(penalty="l1",C=0.1)).fit_transform(data_x,data_y)
    #             elif index == 4:
    #                 data = feature_selection.SelectFromModel(LogisticRegression(penalty="l2",C=0.1)).fit_transform(data_x,data_y)
    #             # 基于树模型，GBDT作为基模型的特征选择
    #             elif index == 5:
    #                 data = feature_selection.SelectFromModel(GradientBoostingClassifier()).fit_transform(data_x,data_y)
    #             # 主成分分析法PCA
    #             elif index == 6:
    #                 data = PCA(n_components=0.7*len(data_x[0])).fit_transform(data_x)
    #     return data


    '''calculate the scores of current code_matrix'''

    def calValue(self, code_matrix, fs_matrix, dataType):
        # estimator = KNeighborsClassifier(n_neighbors=3)
        # estimator = DecisionTreeClassifier()
        estimator = SVC()
        # code_matrix = [[-1,1,-1],[0,-1,-1],[1,-1,1]]
        ecoc_classifier = ECOCClassifier2(estimator, code_matrix.tolist(), fs_matrix)
        if dataType == "validate":
            predict_y = ecoc_classifier.fit_predict(self.train_x, self.train_y, self.validate_x)
            accuracy = ms.f1_score(self.validate_y, predict_y, average="micro")
            # accuracy = ms.accuracy_score(self.validate_y, predict_y)
        elif dataType == "test":
            predict_y = ecoc_classifier.fit_predict(self.train_x, self.train_y, self.test_x)
            accuracy = ms.f1_score(self.test_y, predict_y, average="micro")
            # accuracy = ms.accuracy_score(self.test_y, predict_y)
        return accuracy

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
                index_fs = self.generateIndex(0.2, self.feature_size)
                for k in index_fs:
                    temp = temp_lines[i - 1][j][4][k]
                    temp_lines[i - 1][j][4][k] = temp_lines[i][j][4][k]
                    temp_lines[i][j][4][k] = temp
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
                        index_fs = self.generateIndex(self.pm, self.feature_size)
                        for l in index_fs:
                            if top_lines[i][j][k][l] == 0:
                                top_lines[i][j][k][l] = 1
                            else:
                                top_lines[i][j][k][l] = 0
                            tempInt = top_lines[i][j][k]
                    elif k == 5:
                        tempInt = random.randint(1, 11)
                        while tempInt == top_lines[i][j][k]:
                            tempInt = random.randint(1, 11)
                    top_lines[i][j][k] = tempInt


    '''若某gene生成分类器效果差，则突变该gene的特征选择序列'''
    def mutation4FS(self, gene):
        index = self.generateIndex(self.pm, self.feature_size)  # 挑选特征选择序列中的几位进行突变
        for i in index:
            if gene[4][i] == 1:
                gene[4][i] == 0
            else:
                gene[4][i] = 1
        # 特征选择序列合法性检查
        if 1 not in gene[4]:
            gene[4][random.randint(0,self.feature_size-1)] = 1

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
                print("66666")
                print(line)
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
                    print("77777")
                    print(line)
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
            top_code_matrixs = self.generateSubCodes(code_matrixs, top_lines)
            top_fs_matrixs = self.generateSubFsMatrix(top_lines)

            top_values = []
            for j in range(len(top_code_matrixs)):
                top_values.append(self.calValue(top_code_matrixs[j], top_fs_matrixs[j], "validate"))
            print("top_lines对应的子矩阵分数:")
            print(top_values)

            '''sort top lines according to top_values'''
            self.sort(top_lines, top_code_matrixs, top_values, top_fs_matrixs)
            accuracy.append(max(top_values))
            avg_accuracy.append(sum(accuracy) / len(accuracy))
            w_sheet2.write(Index*2-1, i+1, max(top_values))

            print("最大值:")
            print(max(top_values))
            print(top_code_matrixs[0])
            print("最好结果所生成的特征矩阵1")
            print(top_fs_matrixs[0])

            test_value = self.calValue(top_code_matrixs[0], top_fs_matrixs[0], "test")
            best_fs_matrixs = top_fs_matrixs[0]
            w_sheet2.write(Index*2, i+1, test_value)

            test_accuracy.append(test_value)
            avg_test_accuracy.append(sum(test_accuracy) / len(test_accuracy))

            distances.append(self.calDistance(top_code_matrixs[0]))

            '''若某gene分数差则变异该gene特征选择矩阵'''
            avg = sum(accuracy) / len(accuracy)
            for i in range(len(top_lines)):
                num = 0
                for j in range(top_code_matrixs[i].shape[1]):
                    genematrixs = []
                    for k in range(self.class_size):
                        genematrixs.append([top_code_matrixs[i][k][j]])
                    score = self.calValue(np.array(genematrixs),np.array([top_fs_matrixs[i][j]]),"validate")
                    if score < avg:
                        self.mutation4FS(top_lines[i][j])
                        num = num + 1

            temp_top_lines = copy.deepcopy(top_lines)
            random.shuffle(temp_top_lines)
            son_top_lines = self.cross(temp_top_lines, self.num_classifier)
            self.mutation(son_top_lines, self.num_classifier)
            son_code_matrixs = self.generateSubCodes(code_matrixs, son_top_lines)
            son_fs_matrixs = self.generateSubFsMatrix(son_top_lines)
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
            # random.shuffle(top_lines)

        # 记录初始矩阵分数
        for i in range(len(code_matrixs)):
            value = self.calValue(code_matrixs[i], best_fs_matrixs, "test")
            w_sheet.write(Index, i + 1, value)

        return accuracy, test_accuracy, avg_accuracy, avg_test_accuracy, distances


''',
    ,'Breast','Cancers','DLBCL',
            'Lung1','Lung2','SRBCT',
            ,'GCM','Leukemia2','Leukemia1','iris',
             'mfeatpix','mfeatzer','mfeatmor',
             'ecoli',
'thyroid', 'vertebral','cmc',
            '''
datasets = [

'glass','wine','dermatology',
'mfeatmor','vehicle','abalone','ecoli',
'vehicle',
'yeast','zoo',
			]
rbook = xlrd.open_workbook('./figure2.xls',formatting_info=False)
wbook = cp.copy(rbook)
w_sheet = wbook.get_sheet(0)
w_sheet2 = wbook.get_sheet(1)
Index = 1
for dataset in datasets:
    for i in range(1):
        Row = dataset + str(i)
        w_sheet.write(Index, 0, Row)
        w_sheet2.write(Index*2-1, 0, Row+'_train')
        w_sheet2.write(Index*2, 0, Row + '_test')
        ##########################################################################
        # DataLoader
        ##########################################################################
        trainfile = "./data/" + str(dataset) + "_train.data"
        testfile = "./data/" + str(dataset) + "_test.data"
        validatefile = "./data/" + str(dataset) + "_validation.data"
        # 每次读入数据时打乱重新读入
        Tool.Shuffle(str(dataset))

        # 其中x为特征空间，y为样本的标签
        train_x, train_y, validate_x, validate_y, instance_size = DataLoader.loadDataset(trainfile, validatefile)
        train_x, train_y, test_x, test_y, instance_size = DataLoader.loadDataset(trainfile, testfile)

        ##########################################################################
        # 配置参数运行程序
        ##########################################################################
        class_size = len(np.unique(np.array(train_y)))
        feature_size = len(train_x[0])
        pop_size = 50  # 种群个体数量
        pc = 0.8
        pm = 0.1
        iteration = 60
        code_pool_size = 25
        ga_top = GA_TOP(class_size, feature_size, pop_size, pc, pm, iteration, code_pool_size,
                        train_x, train_y, validate_x, validate_y, test_x, test_y)

        accuracy, test_accuracy, avg_accuracy, avg_test_accuracy, distances = ga_top.main()
        '''print out the result'''
        #plt.plot(accuracy, label="Validate")
        # plt.plot(avg_accuracy,label = "Va",color = "g")
        #plt.plot(test_accuracy, label="Test", color="r")
        # plt.plot(avg_test_accuracy,label="Ta",color="m")
        #plt.title("top accuracy")
        #plt.legend()

        #filename = "./figures/" + str(dataset) + "/" + str(dataset) + str(i + 1) + ".png"
        #plt.savefig(filename)
        # plt.show()

        #plt.plot(avg_accuracy, label="Validate")
        #plt.plot(avg_test_accuracy, label="Test", color="r")
        #plt.title("avg accuracy")
        #plt.legend()
        # plt.show()

        #plt.plot(distances)
        #filename = "./figures/" + str(dataset) + "/" + str(dataset) + str(i + 1) + "_distance.png"
        #plt.savefig(filename)
        # plt.show()

        recordfilename = "./uci/" + str(dataset) + ".txt"
        recordfile = open(recordfilename, "a")
        print(i, file=recordfile)
        print("ga_top最大值:", file=recordfile)
        print(max(accuracy), file=recordfile)
        print(max(test_accuracy), file=recordfile)
        print("ga_top平均值:", file=recordfile)
        print(sum(accuracy) / len(accuracy), file=recordfile)
        print(sum(test_accuracy) / len(test_accuracy), file=recordfile)
        recordfile.close()
        wbook.save('./figure2.xls')
        Index = Index + 1