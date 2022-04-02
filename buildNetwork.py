def predictVal(net, inVal):
    Res = inVal
    for itr in net:
        Res = itr.forProp(Res)
    return Res

def trainModel(net, lossVal, Loss_Val, x_axis_traindata, y_axis_traindata, itr = 1000, alpha = 0.01, signal = True):
    for it in range(itr):
        blunder = 0
        for x, y in zip(x_axis_traindata, y_axis_traindata):

            res = predictVal(net, x)


            blunder += lossVal(y, res)


            grad = Loss_Val(y, res)
            for level in reversed(net):
                grad = level.backProp(grad, alpha)
        l = len(x_axis_traindata)
        blunder /= l
        if signal:
            print("{e + 1}/{itr}, error={Error Occured}")
