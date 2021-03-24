import csv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):

    if random_state is None:
        random_state = np.random
    # TODO fixme
    exam,fea = x_all_LF.shape
    N = math.ceil(frac_test*exam)

    # np.random.RandomState(random_state)
    temp = random_state.permutation(x_all_LF)
    x_test_NF = temp[0:N,:]
    x_train_MF = temp[N:,:]
    return x_train_MF, x_test_NF



df = pd.read_csv("pima-indians-diabetes.csv")
# print(df)
train_MF, test_NF = split_into_train_and_test(df, frac_test=268/768, random_state=np.random.RandomState(0))
# print(train_MF.shape)
# print(test_NF.shape)

# xTrain = df.drop("HasDiabetes",axis = 1)
xTest = test_NF[:,:-1]
yTest = test_NF[:,-1]
# print(np.count_nonzero(yTest))
xTrain = train_MF[:,:-1]
yTrain = train_MF[:,-1]

# print(xTrain)
# print(yTrain)
# print(xTrain.shape)
# print(yTrain.shape)
# print(xTest.shape)
# print(yTest.shape)
xTrain = (xTrain - np.min(xTrain,axis = 0))/(np.max(xTrain,axis = 0)-np.min(xTrain,axis = 0))
# print(xTrain)
# print(np.max(xTrain,axis = 0))

def predictTest(xTest,weight,yTest):
    row,col = xTest.shape
    dummColumn = np.ones((row,))
    HBatch = np.column_stack((dummColumn, xTest))

    scores = np.dot(HBatch, weight)
    # print(scores)
    prediction = logProb(scores)
    # print(yTest,prediction)
    SSE = np.sum((yTest - np.where(prediction > 0.5, 1, 0)) ** 2)
    return SSE

def sgdLinear(epoch,stepSize,fakeOnlineTrainX,fakeOnlineTrainY,xTest,yTest):
    row,col = fakeOnlineTrainX.shape
    dummColumn = np.ones((row,))
    HBatch = np.column_stack((dummColumn, fakeOnlineTrainX))
    yBatch = fakeOnlineTrainY
    weights = [0 for i in range(col+1)]
    sumLoss = 0
    aveLoss = []
    l2Weights = []
    SSEArr = []
    for t in range(epoch):
        obser = np.random.randint(row)
        sample = HBatch[obser]
        trueY = fakeOnlineTrainY[obser]
        predict = weight@input

        scores = np.dot(sample, weights)
        prediction = logProb(scores)

        thisPredicated = 0 if prediction < 0.5 else 1
        thisL2Loss = (trueY - thisPredicated) ** 2
        sumLoss += thisL2Loss

        if t > 0 and (t % 100 == 0):
            aveLoss.append(sumLoss / t)
            SSEArr.append(predictTest(xTest, weights, yTest))

        errSignal = yBatch[obser] - prediction
        gradient = np.dot(-2*sample.T, errSignal)
        # for j in range
        weights = weights + stepSize * gradient

        thisL2Weights = np.sqrt(np.sum(weights**2))
        l2Weights.append(thisL2Weights)


    return weights,aveLoss,l2Weights,SSEArr

def sgaLogistic(epoch,stepSize,fakeOnlineTrainX,fakeOnlineTrainY,xTest,yTest):
    row,col = fakeOnlineTrainX.shape
    dummColumn = np.ones((row,))
    HBatch = np.column_stack((dummColumn, fakeOnlineTrainX))
    yBatch = fakeOnlineTrainY
    weights = np.array([0 for i in range(col+1)])
    AlltimeWeight = []
    # print(weight.shape)
    sumLoss = 0
    aveLoss = []
    l2Weights = []
    SSEArr = []
    for t in range(epoch):
        obser = np.random.randint(row)
        sample =  HBatch[obser]
        Ind = yBatch[obser]
        # print(sample.shape)
        # return
        scores = np.dot(sample, weights)
        prediction = logProb(scores)

        thisPredicated = 0 if prediction < 0.5 else 1
        thisL2Loss = (Ind - thisPredicated) ** 2
        sumLoss += thisL2Loss

        if t > 0 and (t%100 == 0):
            aveLoss.append(sumLoss /t)
            SSEArr.append(accuray(xTest,weights,yTest,1))
            AlltimeWeight.append(weights)



        errSignal =  Ind - prediction
        gradient = np.dot(sample.T, errSignal)
        # for j in range
        weights = weights + stepSize * gradient

        thisL2Weights = np.sqrt(np.sum(weights ** 2))
        l2Weights.append(thisL2Weights)


    return AlltimeWeight,aveLoss,l2Weights,SSEArr

def logProb(scores):
    return 1/(1+np.exp(-scores))

def accuray(xTest,weights,yTest,Bias):
    if Bias == 1:
        row, col = xTest.shape
        dummColumn = np.ones((row,))
        HBatch = np.column_stack((dummColumn, xTest))
    else:
        HBatch = xTest
    scores = np.dot(HBatch,weights)
    # print(scores)
    prediction = logProb(scores)
    # print(yTest,prediction)
    SSE = np.sum((yTest - np.where(prediction > 0.5, 1, 0)) ** 2)
    # print(SSE)
    return SSE

def Logistic(epoch,stepSize,fakeOnlineTrainX,fakeOnlineTrainY,xTest,yTest):
    row,col = fakeOnlineTrainX.shape
    dummColumn = np.ones((row,))
    HBatch = np.column_stack((dummColumn, fakeOnlineTrainX))
    yBatch = fakeOnlineTrainY
    weights = np.array([0 for i in range(col+1)])
    accArr = []
    for t in range(epoch):
        obser = np.random.randint(row)
        sample = HBatch[obser]

        scores = np.dot(sample,weights)
        prediction = logProb(scores)
        errSignal = yBatch[obser] - prediction
        gradient = np.dot(sample.T, errSignal)
        # for j in range
        weights = weights + stepSize * gradient

        accArr.append(accuray(HBatch, weights, yBatch))

    return weights, accArr
# Little Test
# print(xTrain.shape)
# weights,accurace = Logistic(100000,0.025,xTrain,yTrain,xTest,yTest)
# # print(weights)
# # plt.title("Epoch = 100000, eta = 1e-5")
# plt.plot(accurace)
# plt.xlabel("steps")
# plt.ylabel("SSE")
# plt.show()
# for stepSize in [0.8,1e-3,1-6]:
#     trainedWeightLinear,aveLossLinear, l2WeightsLinear,SSEArrLinear = sgdLinear(100000,0.8,xTrain,yTrain,xTest,yTest)


fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

bestModel = [[],0]
minTestSSE = 1e10
for stepSize in [0.8,1e-3,1e-5]:
    AlltimeWeightLogistic, aveLossLogistic, l2WeightsLogistic, SSEArrLogistic = sgaLogistic(100000,stepSize,xTrain,yTrain,xTest,yTest)

    if np.min(SSEArrLogistic) < minTestSSE:
        minTestSSE = np.min(aveLossLogistic)
        timp100 = np.argmin(aveLossLogistic)
        bestModel[0] =  AlltimeWeightLogistic[timp100]
        bestModel[1] = stepSize

    ax1.plot(aveLossLogistic,label = "eta= "+str(stepSize))
    ax2.plot(l2WeightsLogistic,label = "eta= "+str(stepSize))
    ax3.plot(SSEArrLogistic,label = "eta= "+str(stepSize))

    ax1.set_ylabel("aveLossLogistic")
    ax2.set_ylabel("l2WeightsLogistic")
    ax3.set_ylabel("SSELogistic")
    legend1 = ax1.legend(fontsize='x-large')
    legend2 = ax2.legend(fontsize='x-large')
    legend3 = ax3.legend(fontsize='x-large')


# fig2 = plt.figure()
# ax21 = fig2.add_subplot(131)
# ax22 = fig2.add_subplot(132)
# ax23 = fig2.add_subplot(133)
#
# for stepSize in [0.8,1e-3,1e-5]:
#     WeightLinear,aveLossLinear, l2WeightsLinear,SSEArrLinear = sgaLogistic(100000,stepSize,xTrain,yTrain,xTest,yTest)
#
#     ax21.plot(aveLossLinear,label = "eta= "+str(stepSize))
#     ax22.plot(l2WeightsLinear,label = "eta= "+str(stepSize))
#     ax23.plot(SSEArrLinear,label = "eta= "+str(stepSize))
#
#     ax21.set_ylabel("aveLossLinear")
#     ax22.set_ylabel("l2WeightsLinear")
#     ax23.set_ylabel("SSEArrLinear")
#     legend21 = ax21.legend(fontsize='x-large')
#     legend22 = ax22.legend(fontsize='x-large')
#     legend23 = ax23.legend(fontsize='x-large')
#
plt.show()


