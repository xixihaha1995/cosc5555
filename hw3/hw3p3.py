# x = [
#     [1,24,40000],
#     [1,53,52000],
#     [1,23,25000],
#     [1,25,77000],
#     [1,32,48000],
#     [1,52,110000],
#     [1,22,38000],
#     [1,43,44000],
#     [1,52,27000],
#     [1,48,65000]
#      ]
import matplotlib.pyplot as plt
import numpy as np
x =np.array([
    [24,40000],
    [53,52000],
    [23,25000],
    [25,77000],
    [32,48000],
    [52,110000],
    [22,38000],
    [43,44000],
    [52,27000],
    [48,65000]
     ])
y = [1,0,0,1,1,1,1,0,0,1]
row, col = x.shape
def logProb(scores):
    return 1/(1+np.exp(-scores))
def accuray(HBatch,weights,yTest):
    # row, col = xTest.shape
    # dummColumn = np.ones((row,))
    # HBatch = np.column_stack((dummColumn, xTest))
    scores = np.dot(HBatch,weights)
    # print(scores)
    prediction = logProb(scores)
    # print(yTest,prediction)
    SSE = np.sum((yTest - np.where(prediction > 0.5, 1, 0)) ** 2)
    # print(SSE)
    return SSE
def Logistic(epsilon,stepSize,fakeOnlineTrainX,fakeOnlineTrainY):
    row,col = fakeOnlineTrainX.shape
    dummColumn = np.ones((row,))
    HBatch = np.column_stack((dummColumn, fakeOnlineTrainX))
    yBatch = fakeOnlineTrainY
    weights = np.array([0 for i in range(col+1)])
    accArr = []
    for t in range(epsilon):
        scores = np.dot(HBatch,weights)
        prediction = logProb(scores)
        errSignal = yBatch - prediction
        gradient = np.dot(HBatch.T, errSignal)

        weights = weights + stepSize * gradient

        accArr.append(accuray(HBatch, weights, yBatch))


    # finalScore = np.dot(HBatch,weights)
    # preds =np.round(logProb(finalScore))
    # print(preds)

    return weights,accArr
weights,accurace = Logistic(30000,5e-4,x,y)
# print(weights)
plt.title("Epoch = 30000, eta = 5e-4")
plt.plot(accurace)
plt.xlabel("steps")
plt.ylabel("SSE")
plt.show()


