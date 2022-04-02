import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils




from denseLayer import DenseLayer
from MainCnn import MainCNN
from Update import Update
from level import level
from ActivationFunctions import Sigmoid
from lossCalculation import cross_entropy_func, cross_entropy_full
from buildNetwork import trainModel, predictVal

def dataPreprocess(dim1, dim2, depth):
    zeroSeperator = np.where(dim2 == 0)[0][:depth]
    oneSeperator = np.where(dim2 == 1)[0][:depth]
    allData = np.hstack((zeroSeperator, oneSeperator))
    allData = np.random.permutation(allData)
    dim1= dim1[allData]
    dim2 = dim2[allData]
    ln = len(dim1)
    dim1 = dim1.reshape(ln, 1, 28, 28)
    dim1 = dim1.astype("float32") / 255
    dim2 = np_utils.to_categorical(dim2)
    dim2 = dim2.reshape(len(dim2), 2, 1)
    return dim1, dim2

(trainXdata, trainYdata), (testXdata, testYdata) = mnist.load_data()
trainXdata, trainYdata = dataPreprocess(trainXdata, trainYdata, 50)
testXdata, testYdata = dataPreprocess(testXdata, testYdata, 50)


# neural network
buildNet = [
    MainCNN((1, 28, 28), 3, 5),
    Sigmoid(),
    Update((5, 26, 26), (5 * 26 * 26, 1)),
    DenseLayer(5 * 26 * 26, 100),
    Sigmoid(),
    DenseLayer(50, 2),
    Sigmoid()
]

# train
trainModel(
    buildNet,
    cross_entropy_func,
    cross_entropy_full,
    trainXdata,
    trainYdata,
    itr=20,
    alpha=0.1
)

# test
for dim1, dim2 in zip(testXdata, testYdata):
    output = predictVal(buildNet, dim1)
    print("predicted value: {np.argmax(output)}, original label: {np.argmax(y)}")
