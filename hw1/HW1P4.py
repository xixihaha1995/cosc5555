import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")

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
xTest = df_test.drop("ViolentCrimesPerPop",axis = 1)
# print(xTrain.head())
xTrain.assign(Name="weight0")
xTest.assign(Name="weight0")
rowTrain,colTrain = xTrain.shape
rowTest,colTest = xTest.shape
xTrain["weight0"] = [1] * rowTrain
xTest["weight0"] = [1] * rowTest
column = xTrain.columns


xTrain = xTrain / (np.linalg.norm(xTrain.values, axis=0))
xTest = xTest / (np.linalg.norm(xTest.values, axis=0))
# print(np.linalg.norm(xTrain.values, axis=0))

rowTrain,colTrain = xTrain.shape
rowTest,colTest = xTest.shape
# print(xTrain.head())


# print(xTrain.head())
yTrain = df_train.iloc[:,0]
yTest = df_test.iloc[:,0]
# print(yTrain.head())
# for different lambda, different weights
weightGus = np.random.uniform(low=0,high = 1, size=colTrain)
# print(weightGus.shape)
lamd = 600
weightPro = coorLassoSolver(1e-6,lamd,yTrain.values,xTrain.values,weightGus)
# print(weightPro.shape)
weightNow = copy.deepcopy(weightPro)
# print(xTrain.shape)

weightAll=[weightNow]
lamdAll = [lamd]
for i in range(9):
    lamd = lamd/2
    # print("conver:%s, lambda:%s" % (i,lamd))
    weightNext = coorLassoSolver(1e-6,lamd,yTrain.values,xTrain.values,weightNow)
    weightAll.append(weightNext)
    lamdAll.append(lamd)
    # print("conver:%s"% (i))
    weightNow = copy.deepcopy(weightNext)

# print(weightAll)
# partCode

# for idx, name in enumerate(column):
#     print("idx:%s, name:%s" %(idx,name))
# idx:3, name:agePct12t29
# idx:12, name:pctWSocSec
# idx:39, name:PctKids2Par
# idx:45, name:PctIlleg
# idx:66, name:HousVacant

pltX = [math.log(i) for i in lamdAll]
pltPureX = [i for i in lamdAll]
age = []
pctW = []
pctK = []
pctI = []
hous = []

seTrain = []
seTest = []
nonZero = []
for idx, weight in enumerate(weightAll):
    age.append(weight[3])
    pctW.append(weight[12])
    pctK.append(weight[39])
    pctI.append(weight[45])
    hous.append(weight[66])

    seTrain.append(np.sum((yTrain - xTrain @ weight)**2) / rowTrain)
    seTest.append(np.sum((yTest - xTest @ weight) ** 2) / rowTest)
    nonZero.append(np.count_nonzero(weight))

# print(pltX)
# print(age)
# print(pctW)
# print(pctK)
# print(pctI)
# print(hous)

fig, ax = plt.subplots()
ax.plot(pltX, age, label='agePct12t29')
ax.plot(pltX, pctW, label='pctWSocSec')
ax.plot(pltX, pctK,label='PctKids2Par')
ax.plot(pltX, pctI, label='PctIlleg')
ax.plot(pltX, hous,label='HousVacant')
ax.set_xlabel('log(lambda)')
ax.set_ylabel('coefficients')
legend = ax.legend(fontsize='x-large')
fig.savefig("hw1.4.2.png")


fig2, ax2 = plt.subplots()
ax2.plot(pltX, seTrain, label='squared error train')
ax2.plot(pltX, seTest, label='squared error test')
ax2.set_xlabel('log(lambda)')
ax2.set_ylabel('coefficients')
legend2 = ax2.legend(fontsize='x-large')
fig2.savefig("hw1.4.34.png")


fig3, ax3 = plt.subplots()
ax3.plot(pltPureX, nonZero,label='nonZero')
legend3 = ax3.legend(fontsize='x-large')
fig3.savefig("hw1.4.5.png")


plt.show()





