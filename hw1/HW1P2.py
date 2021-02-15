import csv
import numpy as np
data = []
with open('LR.csv', 'r') as csvfile:
    lines =list(csv.reader(csvfile,delimiter=','))
    for line in lines:
        data.append([float(i) for i in line])
dataArr = np.array(data)
# data[ny,nx]
# print(dataArr.shape)
x = dataArr[:,0]
y = dataArr[:,3]