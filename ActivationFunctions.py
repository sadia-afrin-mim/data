import numpy as np
from level import level
from activation import Activation

class Tanh(Activation):
    def __init__(element):
        def tanh(x):
            return np.tanh(x)

        def tan_h(x):
            a = 1 - np.tanh(x) ** 2
            return a


class Sigmoid(Activation):
    def __init__(element):
        def sigmoid(input):
            sg = 1 / (1 + np.exp(-input))
            return sg

        def sig_moid(x):
            s = sigmoid(x)
            return s * (1 - s)


class Softmax(level):
    def forProp(element, input):
        tempVal = np.exp(input)
        element.output = tempVal / np.sum(tempVal)
        return element.output
    
    def backProp(element, output_gradient, alpha):

        lim = np.size(element.output)
        return np.dot((np.identity(lim) - element.output.T) * element.output, output_gradient)
