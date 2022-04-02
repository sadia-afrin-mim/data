import numpy as np
from scipy import signal
from level import level

class MainCNN(level):
    def __init__(element, inSize, kernelSize, depth):
        in_d, in_h, in_w = inSize
        element.depth = depth
        element.input_shape = inSize
        element.input_depth = in_d
        element.output_shape = (depth, in_h - kernelSize + 1, in_w - kernelSize + 1)
        element.kernels_shape = (depth, in_d, kernelSize, kernelSize)
        element.kernels = np.random.randn(*element.kernels_shape)
        element.biases = np.random.randn(*element.output_shape)

    def forProp(element, inData):
        element.input = inData
        element.output = np.copy(element.biases)
        for i in range(element.depth):
            for j in range(element.input_depth):
                element.output[i] += signal.correlate2d(element.input[j], element.kernels[i, j], "valid")
        return element.output

    def backProp(element, OutgradVector, alpha):
        kernelVar = np.zeros(element.kernels_shape)
        inVar = np.zeros(element.input_shape)

        for i in range(element.depth):
            for j in range(element.input_depth):
                kernelVar[i, j] = signal.correlate2d(element.input[j], OutgradVector[i], "valid")
                inVar[j] += signal.convolve2d(OutgradVector[i], element.kernels[i, j], "full")

        element.kernels -= alpha * kernelVar
        element.biases -= alpha * OutgradVector
        return inVar
