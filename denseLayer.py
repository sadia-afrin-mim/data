import numpy as np
from level import level

class DenseLayer(level):
    def __init__(element, inSize, outSize):
        element.weights = np.random.randn(outSize, inSize)
        element.bias = np.random.randn(outSize, 1)

    def forProp(element, inVal):
        element.input = inVal
        bi_as = element.bias
        return np.dot(element.weights, element.input) + bi_as


    def backProp(element, outVar, alpha):
        varWeight = np.dot(outVar, element.input.T)
        varInput = np.dot(element.weights.T, outVar)
        element.weights -= alpha * varWeight
        element.bias -= alpha * outVar
        return varInput
