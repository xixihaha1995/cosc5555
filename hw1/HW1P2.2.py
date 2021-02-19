import csv
import numpy as np
from numpy.linalg import inv

data = []
with open('LR.csv', 'r') as csvfile:
    lines = list(csv.reader(csvfile, delimiter=','))
    for line in lines:
        data.append([float(i) for i in line])
dataArr = np.array(data)
row, col = dataArr.shape
# data[ny,nx]
x = dataArr[:, 0:3]
y = dataArr[:, 3]
# print(type(y))
# print(x)

dummColumn = np.ones((row,))
H = np.column_stack((dummColumn, x))
# print(H)

weightA = np.matmul(inv(np.matmul(np.transpose(H), H)), np.matmul(np.transpose(H), y))
yOne = np.matmul(weightA, [1,1, 1, 1])
yTwo = np.matmul(weightA, [1,2, 0, 4])
yThree = np.matmul(weightA, [1,3,2,1])
# print("weightA:%s, yone:%s, ytwo:%s, ythree:%s"%(weightA,yOne,yTwo,yThree))
# weightA:[ 5.31416717 -2.00371927  0.53256334 -0.26560187], yone:3.577409374298066, ytwo:0.24432117283603283, ythree:0.10253417289227507

def crossValidation(num_folds,H,y):
    X_train_folds = np.array_split(H, num_folds)
    y_train_folds = np.array_split(y, num_folds)

    geneError = []
    for i in range(num_folds):
        X_train_cv = np.vstack(X_train_folds[0:i] + X_train_folds[i + 1:])
        y_train_cv = np.hstack(y_train_folds[0:i] + y_train_folds[i + 1:])
        X_valid_cv = X_train_folds[i]
        y_valid_cv = y_train_folds[i]

        obs= y_valid_cv.shape

        weightA = np.matmul(inv(np.matmul(np.transpose(X_train_cv), X_train_cv)), np.matmul(np.transpose(X_train_cv), y_train_cv))
        yCap = [np.matmul(weightA, item) for item in X_valid_cv]
        geneError.append(np.sum((yCap - y_valid_cv) ** 2))
    generationErrorA = np.sum(geneError) / num_folds/obs
    return generationErrorA

geneErr = crossValidation(5,H,y)
# print(geneErr)
# [5.55476927]

