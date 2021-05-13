import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import rand
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
    # print(np.count_nonzero(x_all_LF[:,-1]))
    print(x_all_LF.shape)
    print(np.count_nonzero(x_all_LF.iloc[:,-1]) /x_all_LF.shape[0] )

    # print(x_train_MF)
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
        # obser = np.random.randint(row)
        obser = t % row
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

class logisticRegression:
    def fit(self,X,y):
        loss = []
        weights = n
    def sigmod(self,z):
        return 1 / (1+np.e**(-z))
    def costFunction(self,X,y,weights):
        z = np.dot(X,weights)
        # prediction = self.sigmod(z)
        # SSE = np.sum((y - np.where(prediction > 0.5, 1, 0)) ** 2)
        predict1 = y* np.log(self.sigmod(z))
        predict0 = (1-y)*np.log(1-self.sigmod(z))
        return -np.sum(predict0+predict1) / len(X)


def main():
    # print("original data")
    bank = pd.read_csv("bank.csv", delimiter=';')
    # print(bank.head())
    # print("after oneHotEncoding")
    df = oneHotEnc(bank)
    df.rename(columns={0: 'y'}, inplace=True)
    # print(type(df))
    # print(df.head())

    train_MF, test_NF = split_into_train_and_test(df, frac_test=0.3, random_state=np.random.RandomState(0))
    xTest = test_NF[:, :-1]
    yTest = test_NF[:, -1]

    print(type(xTest))
    # print(np.count_nonzero(yTest))
    xTrain = train_MF[:, :-1]
    yTrain = train_MF[:, -1]

    xTrain = (xTrain - np.min(xTrain, axis=0)) / (np.max(xTrain, axis=0) - np.min(xTrain, axis=0))
    xTest = (xTest - np.min(xTest, axis=0)) / (np.max(xTest, axis=0) - np.min(xTest, axis=0))


if __name__ == "__main__":
    main()


