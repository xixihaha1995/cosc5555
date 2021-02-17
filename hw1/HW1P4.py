import pandas as pd
import numpy as np
df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")
row,col = df_train.shape
# print(type(row))
# print(row)
# print(df_train.iloc[:,0])
# the first column
def softThreshold(rho,lamd):
    if rho < -lamd/2:
        return rho + lamd/2
    elif rho > lamd /2:
        return rho - lamd/2
    return 0

def coorLassoSolver(converg, lamd, responY, hX,weight):
    l1norm = 10
    m,n = hX.shape
    inputX = hX/(np.linalg.norm(hX, axis= 0))
    yPred =inputX  @ weight
    while l1norm > converg:
        weightNext = weight[:]
        for j in range(n):
            # print(inputX.shape)
            hJ = inputX[:,j].reshape(-1,1)
            # print(hJ.shape)
            rho = np.sum(hJ *(responY - yPred + weight[j]*hJ[j]))
            weightNext[j] = softThreshold(rho,lamd)
        diff = np.abs(weight-weightNext)
        l1norm =max(diff)
        print(l1norm)
    return weight.flatten()


xTrain = df_train.drop("ViolentCrimesPerPop",axis = 1)
xTrain.assign(Name = "weight0")
xTrain["weight0"] =[1]*row
# print(xTrain.head())
yTrain = df_train.iloc[:,0]
# print(yTrain.head())
# for different lambda, different weights
weightInit = np.random.uniform(low=0,high = 1, size=col)
# print(weightInit.shape)
lamd = 600
# print(xTrain.shape)
weight = coorLassoSolver(1e-6,lamd,yTrain.values,xTrain.values,weightInit)
print(weight)





