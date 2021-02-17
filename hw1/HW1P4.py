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
    yPred = weight @ inputX
    while l1norm > converg:
        l1norm = 0
        for j in range(n):
            print(inputX.shape)
            hJ = inputX[:,j].reshape(-1,1)
            print(hJ.shape)
            rho = hJ @ (responY - yPred + weight[j]*hJ)
            oldWeight = weight[j]
            weight[j] = softThreshold(rho,lamd)
            newWeight  = weight[j]
            if np.abs(oldWeight, newWeight) > l1norm:
                l1norm = np.abs(oldWeight, newWeight)
    return weight.flatten()


xTrain = df_train.drop("ViolentCrimesPerPop",axis = 1)
xTrain.assign(Name = "weight0")
xTrain["weight0"] =[1]*row
# print(xTrain.head())
yTrain = df_train.iloc[:,0]
# print(yTrain.head())
# for different lambda, different weights
weightInit = np.random.uniform(size=row)
# print(weightInit.shape)
lamd = 600
print(xTrain.shape)
weight = coorLassoSolver(1e-6,lamd,yTrain.values,xTrain.values,weightInit)






