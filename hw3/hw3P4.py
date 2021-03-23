import csv
import numpy as np
import pandas as pd
import math

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

print(xTrain)
print(yTrain)
# print(xTrain.shape)
# print(yTrain.shape)
# print(xTest.shape)
# print(yTest.shape)
xTrain = (xTrain - np.min(xTrain,axis = 0))/(np.max(xTrain,axis = 0)-np.min(xTrain,axis = 0))
# print(xTrain)
# print(np.max(xTrain,axis = 0))
def sgdLinear(epoch,stepSize,fakeOnlineTrainX,fakeOnlineTrainY):
    row,col = fakeOnlineTrainX.shape
    dummColumn = np.ones((row,))
    HBatch = np.column_stack((dummColumn, fakeOnlineTrainX))
    weight = [0 for i in range(col+1)]
    for t in range(epoch):
        obser = np.random.randint(row)
        input = HBatch[obser]
        trueY = fakeOnlineTrainY[obser]
        predict = weight@input

        weight = weight - stepSize*(-2*input.T*( trueY-predict ))

    return weight

def sgaLogistic(epoch,stepSize,fakeOnlineTrainX,fakeOnlineTrainY):
    row,col = fakeOnlineTrainX.shape
    dummColumn = np.ones((row,))
    HBatch = np.column_stack((dummColumn, fakeOnlineTrainX))
    weight = [0 for i in range(col+1)]
    for t in range(epoch):
        obser = np.random.randint(row)
        sample =  HBatch[obser]
        Ind = fakeOnlineTrainY[obser]
        P =1/(1+np.exp(-1*(weight@sample)))
        for j in range(col+1):
            hjFeature = sample[j]
            derivative = hjFeature*(Ind - P)
            weight = weight +  stepSize*derivative

    return weight

trainedWeightLinear = sgdLinear(100,0.8,xTrain,yTrain)
trainedWeightLogistic = sgaLogistic(100,0.8,xTrain,yTrain)
print(trainedWeightLinear)
print(trainedWeightLogistic)

