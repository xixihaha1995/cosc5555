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

def thisGroupAndError(data):
    oneTarget = data[:,-1]
    if(np.count_nonzero(oneTarget) == len(oneTarget)/2):
        return len(oneTarget)/2
    if(np.count_nonzero(oneTarget) > len(oneTarget)/2):
        # group = 1
        errCount = len(oneTarget) - np.count_nonzero(oneTarget)
    else:
        # group = 0
        errCount = np.count_nonzero(oneTarget)
    return   errCount

def splitIndexAndError(sortOne,curError):
    # for loop look for where to split
    for i in range(len(sortOne)):
        left = sortOne[:i]
        right = sortOne[i:]
        oneErr = (thisGroupAndError(left)+thisGroupAndError(right)) / len(sortOne)
        if oneErr < curError:
            curError = oneErr
            splitIndex = i
    return curError,splitIndex

def whichFeature(sortOne,sortTwo, curError):
    oneErr, oneIndex = splitIndexAndError(sortOne, curError)
    twoErr, twoIndex = splitIndexAndError(sortTwo, curError)

    if (oneErr < curError) and (oneErr < twoErr):
        #     split on feature one
       return 1, oneErr,oneIndex
    if (twoErr < curError) and (twoErr < twoErr):
        #     split on feature one
        return 2,twoErr, twoIndex
    if (oneErr > curError) and (twoErr > curError):
        return 0, curError, 0

def cutedBranch(sortOne,sortTwo,curError,depth):
    feature,smallerError, index = whichFeature(sortOne,sortTwo,curError)
    if feature == 1:
        cutedBranch(sortOne[:index],sortTwo[:index], smallerError, depth+1)
    if feature == 2:
        cutedBranch(sortOne[:index], sortTwo[:index], smallerError, depth+1)
    if feature == 0:
        return depth, smallerError

def decisionTree(xTrain,yTrain):
    allData = np.c_[xTrain,yTrain]
    sortOne = sorted(allData, key=lambda x: x[0])
    sortTwo = sorted(allData, key=lambda x: x[1])
    curError = thisGroupAndError(allData) / len(allData)
    depth = 0
    return cutedBranch(sortOne,sortTwo,curError,depth)



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
# weights,accurace = Logistic(30000,5e-4,x,y)
# # print(weights)
# plt.title("Epoch = 30000, eta = 5e-4")
# plt.plot(accurace)
# plt.xlabel("steps")
# plt.ylabel("SSE")
# plt.show()
depth, classError = decisionTree(x,y)
print(depth,classError)

