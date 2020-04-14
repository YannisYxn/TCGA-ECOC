# coding: utf-8

""" ECOC Classifier
edit by elfen
any question contact with qq 790599245
"""
import numpy as np
import copy
import warnings

from sklearn.metrics import euclidean_distances


def _check_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if (not hasattr(estimator, "decision_function") and
            not hasattr(estimator, "predict_proba")):
        raise ValueError("The base estimator should implement "
                         "decision_function or predict_proba!")


def check_is_fitted(estimator, attributes):
    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]


def _fit_ternary(estimator, X, y):
    """Fit a single ternary estimator. not offical editing.
        delete item from X and y when y = 0
        edit by elfen.
    """
    X, y = X[y != 0], y[y != 0]  # 去掉0代表的类,这里的编码是二元编码，因此不用去掉
    y[y==-1]=0
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        warnings.warn('only one class')
    else:
        estimator = copy.deepcopy(estimator)
        if X.ndim == 1:
            X = X.reshape(-1,1)
        estimator.fit(X, y)
    return estimator


def is_regressor(estimator):
    """Returns True if the given estimator is (probably) a regressor."""
    return getattr(estimator, "_estimator_type", None) == "regressor"


def _predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if is_regressor(estimator):
        return estimator.predict(X)
    try:
        score = np.ravel(estimator.decision_function(X))
    except (AttributeError, NotImplementedError):
        # probabilities of the positive class
        score = estimator.predict_proba(X)[:, 1]
    return score


def _sigmoid_normalize(X):  # sigmoid函数
    return 1 / (1 + np.exp(-X))*2-1  #三元


class ECOCClassifier2:
    def __init__(self, estimator, code_matrix, fs_matrix):
        self.estimator = estimator  # classifier
        self.code_book_ = code_matrix  # code matrix
        self.code_book_fs = fs_matrix  # features subset for each column in matrix

    def fit(self, X, y):
        _check_estimator(self.estimator)
        if hasattr(self.estimator, "decision_function"):
            self.estimator_type = 'decision_function'  # 分类输出是负无穷到正无穷的连续值
        else:
            self.estimator_type = 'predict_proba'  # 分类输出是[0,1]的连续值

        self.classes_ = np.unique(y)
        classes_index = dict((c, i) for i, c in enumerate(self.classes_))
        Y = np.array([self.code_book_[classes_index[y[i]]] for i in range(X.shape[0])], dtype=np.int)  # 分类输出

        # try:
        #     self.estimators_ = [_fit_ternary(self.estimator, X[:, self.code_book_fs[i]], Y[:, i]) for i in
        #                         range(Y.shape[1])]
        # except:
        #     print i
        #print(len(self.code_book_fs))
        #print(len(self.code_book_))
        self.estimators_ = [_fit_ternary(self.estimator, X[:, self.code_book_fs[i]], Y[:, i]) for i in range(Y.shape[1])]  # 所有分类器
        return self

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        Y = np.array(
            [_predict_binary(self.estimators_[i], X[:, self.code_book_fs[i]]) for i in range(len(self.estimators_))]).T
        if self.estimator_type == 'decision_function':
            Y = _sigmoid_normalize(Y)  # 采用sigmoid函数是因为 decision_function 产生的结果的范围是负无穷到正无穷
        pred = euclidean_distances(Y, self.code_book_).argmin(axis=1)
        return self.classes_[pred]

    def validate(self, X):
        check_is_fitted(self, 'estimators_')

        Y = np.array(
            [_predict_binary(self.estimators_[i], X[:, self.code_book_fs[i]]) for i in range(len(self.estimators_))]).T
        if self.estimator_type == 'decision_function':
            Y = _sigmoid_normalize(Y)

        dis = euclidean_distances(Y, self.code_book_)
        pred = dis.argmin(axis=1)
        # pred = euclidean_distances(Y, self.code_book_).argmin(axis=1)
        return self.classes_[pred], dis, Y

    def fit_predict(self, X, y, test_X):
        self.fit(X, y)
        return self.predict(test_X)
