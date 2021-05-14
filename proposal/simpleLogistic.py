import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import rand
from sklearn import preprocessing
from sklearn import metrics, svm
from sklearn.metrics import plot_confusion_matrix, precision_score
from collections import Counter



def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):

    if random_state is None:
        random_state = np.random

    exam,fea = x_all_LF.shape
    N = math.ceil(frac_test*exam)

    # np.random.RandomState(random_state)
    temp = random_state.permutation(x_all_LF)
    x_test_NF = temp[0:N,:]
    x_train_MF = temp[N:,:]
    # print(np.count_nonzero(x_all_LF[:,-1]))
    # print(x_all_LF.shape)
    # print(np.count_nonzero(x_all_LF.iloc[:,-1]) /x_all_LF.shape[0] )

    # print(x_train_MF)
    return x_train_MF, x_test_NF
def oneHotEnc(bank):
    for column in bank:
        if column == 'y':
            temp = bank.y.astype('category').cat.codes
            # print(type(temp))
        else:
            if bank[column].dtypes == object:
                temp = pd.get_dummies(bank[column], prefix=column)
            else:
                temp =  bank[column]
        try:
            # s.append(temp)
            s = pd.concat([s, temp], axis=1)
        except NameError:
            s = pd.DataFrame(data=temp)

    s.rename(columns={0: 'y'}, inplace=True)
    return s


def labelEncoding(bank):
    # le = preprocessing.LabelEncoder()
    for column in bank:
        if bank[column].dtypes == object:
            # le.fit(bank[column])
            # temp = le.transform(bank[column])
            # bank[column].astype('category')
            temp =bank[column].astype('category').cat.codes
            # print(le.transform(bank[column]))
            # temp = bank.y.astype('category').cat.codes
            # print(temp)
        else:
            temp =  bank[column]
        try:
            # s.append(temp)
            s = pd.concat([s, temp], axis=1)
        except NameError:
            s = pd.DataFrame(data=temp)

        s.rename(columns={0: column}, inplace=True)
    # print(s)
    return s



class logisticRegression:
    def fit(self,X,y, epoch = 2e3, lr = 5e-2):
        loss = []
        weights = rand(X.shape[1])
        N = len(X)

        for _ in range(int(epoch)):
            y_hat = self.sigmod(np.dot(X,weights))
            weights += lr*np.dot(X.T, y-y_hat) / N
            loss.append(self.costFunction(X,y,weights))
        self.weights = weights
        self.loss = loss

    def sigmod(self,z):
        return 1 / (1+np.e**(-z))
    def costFunction(self,X,y,weights):
        z = np.dot(X,weights)
        # prediction = self.sigmod(z)
        # SSE = np.sum((y - np.where(prediction > 0.5, 1, 0)) ** 2)
        predict1 = y* np.log(self.sigmod(z))
        predict0 = (1-y)*np.log(1-self.sigmod(z))
        return -np.sum(predict0+predict1) / len(X)

    def predict(self,X):
        z = np.dot(X,self.weights)
        return [1 if i > 0.5 else 0 for i in self.sigmod(z)]

def imbalanced(data):
    N = data.index.size
    dataWithOne = data.loc[data["y"] == 1]
    multiInt =(N - dataWithOne.index.size) // dataWithOne.index.size
    for _ in range(multiInt):
        data = data.append(dataWithOne)
    # print(data.index)
    # print(data.loc[data["y"] == 1].index)
    return data

def bestCustom(xTrain, yTrain, yTest, xTest):

    customModel = logisticRegression()
    maxScore = np.float('-inf')
    bestLR = 0
    for lr in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:
        customModel.fit(xTrain,yTrain,1E4, lr)
        score = precision_score(yTest, customModel.predict(xTest))
        if score > maxScore:
            bestLR = lr
            maxScore = score

    return bestLR


def main():
    # print("original data")
    bank = pd.read_csv("bank.csv", delimiter=';')
    # print(bank.head())
    # print("after oneHotEncoding")
    df = oneHotEnc(bank)
    # df = labelEncoding(bank)
    # print(df.head())
    # print(df.columns)
    # print(dfOnehot.head())
    # print(dfOnehot.columns)

    df = imbalanced(df)
    # print(type(df))
    # print(df.head())

    train_MF, test_NF = split_into_train_and_test(df, frac_test=0.3, random_state=np.random.RandomState(0))
    xTest = test_NF[:, :-1]
    yTest = test_NF[:, -1]

    # print(np.count_nonzero(yTest))
    xTrain = train_MF[:, :-1]
    yTrain = train_MF[:, -1]

    xTrain = (xTrain - np.min(xTrain, axis=0)) / (np.max(xTrain, axis=0) - np.min(xTrain, axis=0))
    xTest = (xTest - np.min(xTest, axis=0)) / (np.max(xTest, axis=0) - np.min(xTest, axis=0))


    customModel = logisticRegression()
    # bestLr = bestCustom(xTrain,yTrain,yTest,xTest)

    customModel.fit(xTrain,yTrain,1E3)
    clf = svm.SVC(random_state=0)
    clf.fit(xTrain,yTrain)
    cmClf = metrics.confusion_matrix(yTest, clf.predict(xTest))
    cmCustom = metrics.confusion_matrix(yTest, customModel.predict(xTest))
    dispClf =metrics.ConfusionMatrixDisplay(confusion_matrix=cmClf)
    dispCustom = metrics.ConfusionMatrixDisplay(confusion_matrix=cmCustom)
    dispClf.plot()
    dispClf.figure_
    dispCustom.plot()

    print(precision_score(yTest, clf.predict(xTest)))
    plt.show()


if __name__ == "__main__":
    main()


