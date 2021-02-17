import pandas as pd
import numpy as np
df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")
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
            hJ = inputX[:,j].reshape(-1,1)
            rho = hJ @ (responY - yPred + weight[j]*hJ)
            weight[j] = softThreshold(rho,lamd)
    return weight.flatten()




