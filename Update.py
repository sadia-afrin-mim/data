import numpy as np
from level import level

class Update(level):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forProp(self, input):
        return np.reshape(input, self.output_shape)

    def backProp(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
