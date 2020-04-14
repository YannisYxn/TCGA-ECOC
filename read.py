"""
this model define some method to read data from files
"""

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler


def read_multi_yeast_dataset(path):
    df = pd.read_csv(path, header=None, low_memory=False)
    df_values = df.values
    col_num = df_values.shape[1]
    row_num = df_values.shape[0]
    col_heads = df_values[0, :]

    for i in range(1, row_num):
        for j in range(0, col_num):
            df_values[i, j] = float(df_values[i, j])

    attr_count = 0
    for head in col_heads:
        head = str(head)
        if head.startswith('Class'):
            break
        attr_count += 1
    class_num = col_num - attr_count
    data = df_values[1:, 0:col_num - class_num]
    label = df_values[1:, col_num - class_num: col_num]

    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            label[i][j] = int(label[i][j])

    attributes = df_values[0, 0:col_num - class_num]
    classes = df_values[0, col_num - class_num: col_num]
    return data, label, attributes, classes


def read_multi_emotions_dataset(path):
    df = pd.read_csv(path, header=None, low_memory=False)
    df_values = df.values
    col_num = df_values.shape[1]
    row_num = df_values.shape[0]
    col_heads = df_values[0, :]

    for i in range(1, row_num):
        for j in range(0, col_num):
            df_values[i, j] = float(df_values[i, j])

    attr_count = 0
    for head in col_heads:
        head = str(head)
        if head.startswith('Amazed'):
            break
        attr_count += 1
    class_num = col_num - attr_count
    data = df_values[1:, 0:col_num - class_num]
    label = df_values[1:, col_num - class_num: col_num]

    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            label[i][j] = int(label[i][j])

    attributes = df_values[0, 0:col_num - class_num]
    classes = df_values[0, col_num - class_num: col_num]
    return data, label, attributes, classes


def read_multi_medical_dataset(path):
    df = pd.read_csv(path, header=None, low_memory=False)
    df_values = df.values
    col_num = df_values.shape[1]
    row_num = df_values.shape[0]
    col_heads = df_values[0, :]

    for i in range(1, row_num):
        for j in range(0, col_num):
            df_values[i, j] = int(df_values[i, j])

    attr_count = 0
    for head in col_heads:
        head = str(head)
        if head.startswith('Class'):
            break
        attr_count += 1
    class_num = col_num - attr_count
    data = df_values[1:, 0:col_num - class_num]
    label = df_values[1:, col_num - class_num: col_num]
    attributes = df_values[0, 0:col_num - class_num]
    classes = df_values[0, col_num - class_num: col_num]
    return data, label, attributes, classes


def read_multi_enron_dataset(path):
    df = pd.read_csv(path, header=None, low_memory=False)
    df_values = df.values
    col_num = df_values.shape[1]
    row_num = df_values.shape[0]
    col_heads = df_values[0, :]

    for i in range(1, row_num):
        for j in range(0, col_num):
            df_values[i, j] = int(df_values[i, j])

    attr_count = 0
    for head in col_heads:
        head = str(head)
        if head.startswith('Class'):
            break
        attr_count += 1
    class_num = col_num - attr_count
    data = df_values[1:, 0:col_num - class_num]
    label = df_values[1:, col_num - class_num: col_num]
    attributes = df_values[0, 0:col_num - class_num]
    classes = df_values[0, col_num - class_num: col_num]
    return data, label, attributes, classes


def read_genbase_dataset(path):
    df = pd.read_csv(path, header=None, low_memory=False)
    df_values = df.values
    df_values = df_values[:, 1:]
    col_num = df_values.shape[1]
    row_num = df_values.shape[0]
    col_heads = df_values[0, :]

    for i in range(1, row_num):
        for j in range(0, col_num):
            if df_values[i, j] == 'NO':
                df_values[i, j] = 0
            elif df_values[i, j] == 'YES':
                df_values[i, j] = 1
            else:
                df_values[i, j] = int(df_values[i, j])

    attr_count = 0
    for head in col_heads:
        head = str(head)
        if head.startswith('PDOC'):
            break
        attr_count += 1
    class_num = col_num - attr_count
    data = df_values[1:, 0:col_num - class_num]
    label = df_values[1:, col_num - class_num: col_num]
    attributes = df_values[0, 0:col_num - class_num]
    classes = df_values[0, col_num - class_num: col_num]
    return data, label, attributes, classes


def read_birds(path):
    df = pd.read_csv(path, header=None, low_memory=False)
    df_values = df.values
    col_num = df_values.shape[1]
    row_num = df_values.shape[0]
    col_heads = df_values[0, :]

    for i in range(1, row_num):
        for j in range(0, col_num):
            df_values[i, j] = float(df_values[i, j])

    attr_count = 0
    for head in col_heads:
        head = str(head)
        if head.startswith('\'Brown'):
            break
        attr_count += 1
    class_num = col_num - attr_count
    data = df_values[1:, 0:col_num - class_num]
    label = df_values[1:, col_num - class_num: col_num]

    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            label[i][j] = int(label[i][j])

    attributes = df_values[0, 0:col_num - class_num]
    classes = df_values[0, col_num - class_num: col_num]
    return data, label, attributes, classes


def read_matrix(path, feature_type=str, ignore_col_num=0, ignore_row_num=0):
    df = pd.read_csv(path, header=None)
    df_values = df.values
    row_num = df_values.shape[0]
    col_num = df_values.shape[1]
    data = df_values[ignore_row_num: row_num, ignore_col_num: col_num]
    if feature_type is not str:  # change the data type to the input type
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = feature_type(data[i][j])
    return data


def read_UCI_Dataset(path):
    """
    to read UCI data set from file
    :param path: path of file
    :return: data, label
    """
    df = pd.read_csv(path, header=None)
    df_values = df.values
    col_num = df_values.shape[1]
    data = df_values[:, 0:col_num - 1]
    label = df_values[:, col_num - 1]
    return data, label


def read_Microarray_Dataset(path):
    """
    to read Microarray data set from file
    :param path: path of file
    :return:
    """
    pattern = re.compile(r'(\w+)(\.)*.*')
    df = pd.read_csv(path)
    df_columns = np.array([pattern.match(col).group(1) for col in df.axes[1]])
    df_values = df.values
    data = df_values.T
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    label = df_columns
    return data, label


    # data, label = read_Microarray_Dataset(r'E:\XMU\Data\Micro array\microarray\Breast_train.csv')
