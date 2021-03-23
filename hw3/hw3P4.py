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
    predict = HBatch@weight
    prob = 1 / (1 + np.exp(-predict))
    # print(prob)
    # print(np.where(prob > 0.5, 1, 0))
    SSE = np.sum((yTest -  np.where(prob > 0.5, 1, 0))**2)
    return SSE

def sgdLinear(epoch,stepSize,fakeOnlineTrainX,fakeOnlineTrainY,xTest,yTest):
    row,col = fakeOnlineTrainX.shape
    dummColumn = np.ones((row,))
    HBatch = np.column_stack((dummColumn, fakeOnlineTrainX))
    weight = [0 for i in range(col+1)]
    sumLoss = 0
    aveLoss = []
    l2Weights = []
    SSEArr = []
    for t in range(epoch):
        obser = np.random.randint(row)
        input = HBatch[obser]
        trueY = fakeOnlineTrainY[obser]
        predict = weight@input

        Probility = 1 / (1 + np.exp(-predict))
        thisPredicated = 0 if Probility < 0.5 else 1
        thisL2Loss = (trueY - thisPredicated) ** 2
        sumLoss += thisL2Loss

        if t >0 and (t%100 == 0):
            aveLoss.append(sumLoss / t)
            SSEArr.append(predictTest(xTest,weight,yTest))

        # update weights
        weight = weight - stepSize*(-2*input.T*( trueY-predict ))
        thisL2Weights = np.sqrt(np.sum(weight**2))
        l2Weights.append(thisL2Weights)


    return weight,aveLoss,l2Weights,SSEArr

def sgaLogistic(epoch,stepSize,fakeOnlineTrainX,fakeOnlineTrainY):
    row,col = fakeOnlineTrainX.shape
    dummColumn = np.ones((row,))
    HBatch = np.column_stack((dummColumn, fakeOnlineTrainX))
    weight = np.array([0 for i in range(col+1)])
    # print(weight.shape)
    sumLoss = 0
    aveLoss = []
    l2Weights = []
    for t in range(epoch):
        obser = np.random.randint(row)
        sample =  HBatch[obser]
        # print(sample.shape)
        # return
        Ind = fakeOnlineTrainY[obser]
        P =1/(1+np.exp(-1*(weight@sample)))
        if t > 0:
            thisPredicated = 0 if P < 0.5 else 1
            thisL2Loss = (Ind - thisPredicated)**2
            sumLoss += thisL2Loss
            aveLoss.append(sumLoss /t)
        # update weights
        for j in range(col+1):
            hjFeature = sample[j]
            derivative = hjFeature*(Ind - P)
            weight[j] = weight[j] +  stepSize*derivative

        thisL2Weights = np.sqrt(np.sum(weight**2))
        l2Weights.append(thisL2Weights)

    return weight

trainedWeightLinear,aveLossLinear, l2WeightsLinear,SSEArrLinear = sgdLinear(100000,1e-6,xTrain,yTrain,xTest,yTest)
fig, ax = plt.subplots()
ax.plot(aveLossLinear)
ax.set_ylabel("aveLossLinear")
fig2, ax2 = plt.subplots()
ax2.plot(l2WeightsLinear)
ax2.set_ylabel("l2WeightsLinear")
fig3, ax3 = plt.subplots()
ax3.plot(SSEArrLinear)
ax3.set_ylabel("SSEArrLinear")
plt.show()



# trainedWeightLogistic = sgaLogistic(100000,1e-6,xTrain,yTrain)
# print(trainedWeightLinear)
# print(trainedWeightLogistic)

