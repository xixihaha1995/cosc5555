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

def Logistic(epsilon,stepSize,fakeOnlineTrainX,fakeOnlineTrainY):
    row,col = fakeOnlineTrainX.shape
    dummColumn = np.ones((row,))
    HBatch = np.column_stack((dummColumn, fakeOnlineTrainX))
    yBatch = fakeOnlineTrainY
    weight = np.array([0 for i in range(col+1)])
    partial = [10 for i in range(col+1)]
    breakWhile = 100
    while(breakWhile > epsilon):
        for j in range(col+1):
            partial[j] = 0
            for i in range(row):
                P = 1 / (1 + np.exp(-1 * (weight.T @HBatch[i])))
                hjFeature = HBatch[i,j]
                partial[j] += hjFeature*(yBatch[i] - P)
            weight[j] = weight[j] + stepSize*partial[j]
        breakWhile = abs(np.max(partial))
    return weight
weight = Logistic(1e-2,1e-3,x,y)

