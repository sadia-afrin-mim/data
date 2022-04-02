import numpy as np

def mean_sqaured_error(label, prediction):
    err = label - prediction
    return np.mean(np.power(err, 2))

def mean_squared_error_cross(label, prediction):
    er = (prediction - label)
    return 2 * er / np.size(label)

def cross_entropy_func(label, prediction):
    param = -label * np.log(prediction) - (1 - label) * np.log(1 - prediction)
    return np.mean(param)

def cross_entropy_full(label, prediction):
    return ((1 - label) / (1 - prediction) - label / prediction) / np.size(label)
