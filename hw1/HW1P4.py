import pandas as pd
import numpy as np
import copy
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

def coorLassoSolver(converg, lamd,Y, inputX,*passweight):
    l1norm = 10
    m,n = inputX.shape

    # weight = np.array([i for i in passweight])
    responY = Y.reshape(-1,1)
    weight = np.asarray(passweight).T
    prevWei = copy.deepcopy(weight)
    while l1norm > converg:
        for j in range(n):
            # print(inputX.shape)

            hJ = inputX[:,j].reshape(-1,1)
            yPred = inputX @ weight
            # rho = X_j.T @ (y - y_pred + theta[j] * X_j)
            rho = hJ.T @(responY - yPred+weight[j]*hJ)
            # rho = np.dot( hJ.T,(responY-yPred+weight[j]*hJ))
            weight[j] = softThreshold(rho,lamd)
        diff = np.abs(weight-prevWei)
        l1norm =max(diff)
        prevWei=copy.deepcopy(weight)
        # print(l1norm)
    return weight.flatten()


xTrain = df_train.drop("ViolentCrimesPerPop",axis = 1)
print(xTrain.head())
xTrain = xTrain / (np.linalg.norm(xTrain.values, axis=0))
xTrain.assign(Name="weight0")
xTrain["weight0"] = [1] * row
print(xTrain.head())


# print(xTrain.head())
yTrain = df_train.iloc[:,0]
# print(yTrain.head())
# for different lambda, different weights
weightGus = np.random.uniform(low=0,high = 1, size=col)
# print(weightGus.shape)
lamd = 600
weightPro = coorLassoSolver(1e-6,lamd,yTrain.values,xTrain.values,weightGus)
# print(weightPro.shape)
weightNow = copy.deepcopy(weightPro)
# print(xTrain.shape)

weightAll=[weightNow]
for i in range(9):
    lamd = lamd/2
    print("conver:%s, lambda:%s" % (i,lamd))
    weightNext = coorLassoSolver(1e-6,lamd,yTrain.values,xTrain.values,weightNow)
    weightAll.append(weightNext)
    # print("conver:%s"% (i))
    weightNow = copy.deepcopy(weightNext)

print(weightAll)





