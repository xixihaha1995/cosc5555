import csv
import numpy as np
from numpy.linalg import inv
data = []
with open('LR.csv', 'r') as csvfile:
    lines =list(csv.reader(csvfile,delimiter=','))
    for line in lines:
        data.append([float(i) for i in line])
dataArr = np.array(data)
# data[ny,nx]
x = dataArr[:,0]
y = dataArr[:,3]
print(type(y))

dummColumn = np.ones((100,))
H = np.column_stack((dummColumn,x))

# print(x)

weightA = np.matmul(inv(np.matmul(np.transpose(H),H)),np.matmul(np.transpose(H),y))
yOne = np.matmul(weightA,[1,1])
yTwo = np.matmul(weightA,[1,2])
yThree = np.matmul(weightA,[1,3])
# print("weightA:%s, yone:%s, ytwo:%s, ythree:%s"%(weightA,yOne,yTwo,yThree))
# weightA:[ 5.92794892 -2.03833663], yone:3.8896122844108025, ytwo:1.851275650759999, ythree:-0.18706098289080408

#CV for 5 fold
geneError = []
for i in range(5):
    HSlice = H[20*i:20*(i+1),:]
    ySlice = y[20*i:20*(i+1)]
    yCap = [np.matmul(weightA,item) for item in HSlice]
    geneError.append(np.sum((yCap - ySlice )**2))
generationErrorA = np.sum(geneError) /5
# print(generationErrorA)
