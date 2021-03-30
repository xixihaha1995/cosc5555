import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import metrics
from collections import Counter



def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):

    if random_state is None:
        random_state = np.random

    exam,fea = x_all_LF.shape
    N = math.ceil(frac_test*exam)

    # np.random.RandomState(random_state)
    temp = random_state.permutation(x_all_LF)
    x_test_NF = temp[0:N,:]
    x_train_MF = temp[N:,:]
    return x_train_MF, x_test_NF
def oneHotEnc(bank):
    for column in bank:
        if column == 'y':
            temp = bank.y.astype('category').cat.codes
            # print(type(temp))
        else:
            if bank[column].dtypes == object:
                temp = pd.get_dummies(bank[column], prefix=column)
            else:
                temp =  bank[column]
        try:
            # s.append(temp)
            s = pd.concat([s, temp], axis=1)
        except NameError:
            s = pd.DataFrame(data=temp)
    return s

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
    # if np.count_nonzero(np.where(prediction > 0.5, 1, 0)) > 0:
    #     # print("I got 1")
    return SSE

def predicted(xTest,weights,Bias):
    if Bias == 1:
        row, col = xTest.shape
        dummColumn = np.ones((row,))
        HBatch = np.column_stack((dummColumn, xTest))
    else:
        HBatch = xTest
    scores = np.dot(HBatch,weights)
    prediction = logProb(scores)

    return np.where(prediction > 0.5, 1, 0)



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

bank = pd.read_csv("bank.csv",delimiter=';')
df = oneHotEnc(bank)
df.rename(columns = {0:'y'}, inplace = True)

train_MF, test_NF = split_into_train_and_test(df, frac_test=0.3, random_state=np.random.RandomState(0))
xTest = test_NF[:,:-1]
yTest = test_NF[:,-1]
# print(np.count_nonzero(yTest))
xTrain = train_MF[:,:-1]
yTrain = train_MF[:,-1]
# print(xTrain)
# print(yTrain)
# minMax normalization
xTrain = (xTrain - np.min(xTrain,axis = 0))/(np.max(xTrain,axis = 0)-np.min(xTrain,axis = 0))
xTest = (xTest - np.min(xTest,axis = 0))/(np.max(xTest,axis = 0)-np.min(xTest,axis = 0))
# print(xTrain)
# print(Counter(yTest))
# print(np.count_nonzero(yTest))
#
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

bestModel = [[],0,0,0]
minTestSSE = 1e10
for stepSize in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:
    AlltimeWeightLogistic, aveLossLogistic, l2WeightsLogistic, SSEArrLogistic = sgaLogistic(100000,stepSize,xTrain,yTrain,xTest,yTest)

    if np.min(SSEArrLogistic) < minTestSSE:
        minTestSSE = np.min(aveLossLogistic)
        timp100 = np.argmin(aveLossLogistic)
        bestModel[0] =  AlltimeWeightLogistic[timp100]
        bestModel[1] = stepSize
        bestModel[2] = timp100*100
        bestModel[3] = minTestSSE

    ax1.plot(aveLossLogistic,label = "eta= "+str(stepSize))
    ax2.plot(l2WeightsLogistic,label = "eta= "+str(stepSize))
    ax3.plot(SSEArrLogistic,label = "eta= "+str(stepSize))

    ax1.set_ylabel("aveLossLogistic")
    ax2.set_ylabel("l2WeightsLogistic")
    ax3.set_ylabel("SSELogistic")
    legend1 = ax1.legend(fontsize='x-large')
    legend2 = ax2.legend(fontsize='x-large')
    legend3 = ax3.legend(fontsize='x-large')



# plt.show()
pred = predicted(xTest,bestModel[0],1)
fpr, tpr, thresholds = metrics.roc_curve(yTest, pred)
print(metrics.auc(fpr, tpr))



