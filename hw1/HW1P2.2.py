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
def crossValidation(num_folds,H,y):
    X_train_folds = np.array_split(H, num_folds)
    y_train_folds = np.array_split(y, num_folds)

    geneError = []
    for i in range(num_folds):
        X_train_cv = np.vstack(X_train_folds[0:i] + X_train_folds[i + 1:])
        y_train_cv = np.hstack(y_train_folds[0:i] + y_train_folds[i + 1:])
        X_valid_cv = X_train_folds[i]
        y_valid_cv = y_train_folds[i]

        weightA = np.matmul(inv(np.matmul(np.transpose(X_train_cv), X_train_cv)), np.matmul(np.transpose(X_train_cv), y_train_cv))
        yCap = [np.matmul(weightA, item) for item in X_valid_cv]
        geneError.append(np.sum((yCap - y_valid_cv) ** 2))
    generationErrorA = np.sum(geneError) / num_folds
    return generationErrorA

geneErr = crossValidation(5,H,y)
print(geneErr)

# weightA = np.matmul(inv(np.matmul(np.transpose(H), H)), np.matmul(np.transpose(H), y))
# yOne = np.matmul(weightA, [1, 1])
# yTwo = np.matmul(weightA, [1, 2])
# yThree = np.matmul(weightA, [1, 3])
# # print("weightA:%s, yone:%s, ytwo:%s, ythree:%s"%(weightA,yOne,yTwo,yThree))
# # weightA:[ 5.92794892 -2.03833663], yone:3.8896122844108025, ytwo:1.851275650759999, ythree:-0.18706098289080408
# # partA up
#
# # CV for 5 fold
# geneError = []
# for i in range(5):
#     HSlice = H[20 * i:20 * (i + 1), :]
#     ySlice = y[20 * i:20 * (i + 1)]
#     yCap = [np.matmul(weightA, item) for item in HSlice]
#     geneError.append(np.sum((yCap - ySlice) ** 2))
# generationErrorA = np.sum(geneError) / 5
# # print(generationErrorA) 39.39972330162378
# # partB up
#
#
# weight = [weightA]
# polynomial = [2, 3, 4, 5]
#
# for poly in polynomial:
#     HPoly = H[:]
#     for p in range(2, poly + 1):
#         HPoly = np.column_stack((HPoly, x ** p))
#
#     weightPoly = np.matmul(inv(np.matmul(np.transpose(HPoly), HPoly)),
#                            np.matmul(np.transpose(HPoly), y))
#     weight.append(weightPoly)
# # print(weight)
# # [array([ 5.92794892, -2.03833663]), array([ 3.27369294,  1.98367369, -1.00550258]), array([ 3.27636183e+00,  1.97545996e+00, -1.00034312e+00, -8.59910796e-04]), array([ 3.26318505,  2.0444677 , -1.0788899 ,  0.02978751, -0.00383093]), array([ 3.27845696,  1.92128111, -0.85896933, -0.11791391,  0.03781207,
# #        -0.0041643 ])]
# validaX = [1,2,3]
# validYPoly=[[yOne,yTwo,yThree]]
#
# for idx,poly in enumerate(polynomial):
#     yPoly = []
#     for curX in validaX:
#         HPoly = []
#         for p in range(poly+1):
#             HPoly.append(curX**p)
#         yPoly.append(np.matmul(weight[idx+1],HPoly))
#     validYPoly.append(yPoly)
# # print(len(validYPoly))
# validYCap = np.array(validYPoly,dtype=list)
# # print(validYCap)
# # [[3.8896122844108025 1.851275650759999 -0.18706098289080408]
# #  [4.251864056472466 3.2190300049772675 0.17519078917085906]
# #  [4.250618769472329 3.219030004977256 0.17643607617099732]
# #  [4.254719440378921 3.2135661153484136 0.1805367470836643]
# #  [4.256502600356072 3.2135661153679393 0.1787535871174839]]
# # partC up
#
#
# cvGeneErr = [generationErrorA]
# for idx, poly in enumerate(polynomial):
#     tempErr = []
#     HPoly = dummColumn[:]
#     for p in range(1,poly + 1):
#         HPoly = np.column_stack((HPoly, x ** p))
#     HPolyNd = np.array(HPoly)
#     # print(HPolyNd.shape)
#     for i in range(5):
#         HSlice = HPolyNd[20 * i:20 * (i + 1), :]
#         ySlice = y[20 * i:20 * (i + 1)]
#         # print(weight[idx+1])
#         yCap = [np.matmul(weight[idx+1], item) for item in HSlice]
#         tempErr.append(np.sum((yCap - ySlice) ** 2))
#     generationError = np.sum(tempErr) / 5
#     cvGeneErr.append(generationError)
# # print(cvGeneErr)
# # [39.39972330162378, 9.476672230736952, 9.476649284085338, 9.476177987732848, 9.475605485228275]


