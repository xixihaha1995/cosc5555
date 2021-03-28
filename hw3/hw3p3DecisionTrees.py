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

#
# def cutedBranch(sortOne,sortTwo,curError,depth):
#     feature,smallerError, index = whichFeature(sortOne,sortTwo,curError)
#     if feature == 0:
#         return depth, smallerError
#     if feature == 1:
#         newDepthLeft, newErroLeft = cutedBranch(sortOne[:index],sortTwo[:index], smallerError, depth+1)
#         newDepthRight, newErrorRight = cutedBranch(sortOne[index:], sortTwo[index:], smallerError, depth + 1)
#     if feature == 2:
#         newDepth, newError =cutedBranch(sortOne[:index], sortTwo[:index], smallerError, depth+1)
#     return newDepth, newError



def errorCount(data):
    # print(data)
    # print(type(data))
    oneTarget = data[:,-1]
    # print(type(oneTarget) )
    if(np.count_nonzero(oneTarget) == len(oneTarget)/2):
        return len(oneTarget)/2
    if(np.count_nonzero(oneTarget) > len(oneTarget)/2):
        # group = 1
        errCount = len(oneTarget) - np.count_nonzero(oneTarget)
    else:
        # group = 0
        errCount = np.count_nonzero(oneTarget)
    return   errCount

def classificationError(err,data):
    return err/len(data)

def splitIndexAndError(allData,feature):
    if feature == 1:
        sortd = sorted(allData, key=lambda x: x[0])
    if feature == 2:
        sortd = sorted(allData, key=lambda x: x[1])
    # for loop look for where to split
    sort = np.array(sortd)
    splitErrDic = []
    for i in range(len(sort)):
        left = np.array(sort[:i,:])
        right = np.array( sort[i:,:])
        curErr = (errorCount(left)+errorCount(right)) / len(sort)
        splitErrDic.append(curErr)
    return min(splitErrDic), np.argmin(splitErrDic)

def whichFeature(parent, parentErr):
    # feature one possible {split:err}
    # feature two possible {split:err}
    # compare the minimum and parentErr
    # return split, feature, index

    oneErr, oneIndex = splitIndexAndError(parent, 1)
    twoErr, twoIndex = splitIndexAndError(parent, 2)

    if (oneErr < parentErr) and (oneErr <= twoErr):
        #     split on feature one
       return 1, oneIndex
    if (twoErr < parentErr) and (twoErr <= oneErr):
        #     split on feature one
        return 2, twoIndex
    return 0, 0

def recur(parent, depth):
    # this Error
    errNumber = errorCount(parent)
    # classificationError(,parent)
    parentErr = errNumber /len(parent)
    # curErr
    nextFeature, index = whichFeature(parent,parentErr)
    if nextFeature == 0:
        # no split
        return errNumber, depth


    if nextFeature == 1:
    #     split on 1
        sortOne = sorted(parent, key=lambda x: x[0])
        forPrint = (parent[index,0]+parent[index+1,0])/2
        # forPrint = parent[index, 0]
        print("feature:" + str(nextFeature)+", "+str(forPrint))
        leftErrNumber, leftDepth = recur(np.array(sortOne[:index]),depth+1)
        rightErrNumber, rightDepth = recur(np.array(sortOne[index:]),depth+1)
    if nextFeature == 2:
        forPrint = (parent[index,1]+parent[index+1,1])/2
        print("feature:" + str(nextFeature)+", "+str(forPrint))
        sortTwo = sorted(parent, key=lambda x: x[1])
        leftErrNumber, leftDepth = recur(np.array(sortTwo[:index]),depth+1)
        rightErrNumber, rightDepth = recur(np.array(sortTwo[index:]),depth+1)
    errNumber = rightErrNumber+leftErrNumber
    depthReturn = max(leftDepth,rightDepth)
    return errNumber,depthReturn

def decisionTree(xTrain,yTrain):
    allData = np.c_[xTrain,yTrain]
    totalErrNumber,finalDepth = recur(allData,0)
    return totalErrNumber / len(allData), finalDepth




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
classError,depth = decisionTree(x,y)
print(depth,classError)

