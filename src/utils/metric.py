import numpy as np


def acurracy(y_predict, y_test):
    return np.sum(y_predict==y_test)/len(y_predict)

def MSE(y_predict, y_test):
    return np.mean((y_predict-y_test)**2)