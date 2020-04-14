"""
this model define some method to calculate distance
1.hmming_distance
2.euclidean_distance
"""

import numpy as np
import copy


def hamming_distance(x, y, weights=None):
    """
    calculate hamming distance
    :param x: a sample
    :param y: another sample
    :param weights: the weights for each feature
    :return: hamming distance
    """
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    distance = np.sum(((1 - np.sign(x * y)) / 2) * weights)
    return distance


def euclidean_distance(x, y, weights=None):
    """
    calulate euclidean distance
    :param x: a sample
    :param y: another sample
    :param weights: the weights for each feature
    :return: euclidean distance
    """
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    temp_x = copy.deepcopy(x)
    temp_y = copy.deepcopy(y)
    distance = np.sqrt(np.sum(np.power(temp_x - temp_y, 2) * weights))
    return distance


def y_euclidean_distance(x, y, weights=None):
    """
    a different euclidean distance
    :param x: a sample
    :param y: another sample
    :param weights: the weights for each feature
    :return: distance
    """
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    distance = np.sqrt(np.sum(np.abs(y) * np.power(x - y, 2) * weights))
    return distance


def posterior_prob(x, y, weights=None):
    """
    for ML-ECOC, calculate the posterior probability of one row
    :param x: one input vector
    :param y: another vector
    :param weights: 
    :return: prob
    """
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    score = 0
    for i in range(len(x)):
        if x[i] == y[i] and x[i] == 1:
            score += weights[i]
    return score


def positive_similarity(x, y, weights=None):
    """
    for ML_ECOC, calculate the distance between the two label sets
    only positive numbers, which is 1, are counted
    :param x: one input vector
    :param y: another input vector
    :param weights: 
    :return: the distance between x and y, a float number from 0 to 1
    """
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    count_positive = 0
    count_same = 0
    for i in range(len(x)):
        if x[i] == 1 or y[i] == 1:
            count_positive += 1
            if x[i] == y[i]:
                count_same += 1
    # print(count_positive, count_same, np.float(count_same/count_positive))
    return np.float(count_same/count_positive)

