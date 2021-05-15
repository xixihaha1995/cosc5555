import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from numpy.random import rand
from sklearn import preprocessing
from sklearn import metrics, svm
from sklearn.metrics import plot_confusion_matrix, precision_score
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import collections



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
    labels = ['0','1']
    plotY = [x_all_LF.shape[0] - np.count_nonzero(x_all_LF.iloc[:,-1]), np.count_nonzero(x_all_LF.iloc[:,-1])]
    # plt.pie(plotY, labels = labels)
    # plt.suptitle("Distribution of Imbalanced Class")
    # plt.show()
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
            if column == 'month':
                temp = bank[column].astype('category').cat.reorder_categories([ 'jan','feb','mar','apr', 'may', 'jun','jul', 'aug', 'sep','oct', 'nov', 'dec']).cat.codes
            else:
                temp = bank[column].astype('category').cat.codes
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



class CustomlogisticRegression:
    def __init__(self, epoch = None):
        if epoch is None:
            self.epoch = 1e3
        else:
            self.epoch = epoch
    def fit(self,X,y, lr = 5e-2):
        loss = []
        weights = rand(X.shape[1])
        N = len(X)

        for _ in range(int(self.epoch)):
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

    labels = ['0','1']
    plotY = [data.shape[0] - np.count_nonzero(data.iloc[:,-1]), np.count_nonzero(data.iloc[:,-1])]
    plt.pie(plotY, labels = labels)
    plt.suptitle("Distribution of balanced Class")

    return data

def bestCustom(xTrain, yTrain, yTest, xTest):

    customModel = CustomlogisticRegression()
    maxScore = np.float('-inf')
    bestLR = 0
    for lr in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:
        customModel.fit(xTrain,yTrain,1E4, lr)
        score = precision_score(yTest, customModel.predict(xTest))
        if score > maxScore:
            bestLR = lr
            maxScore = score

    return bestLR
def multiConfusionPlot(X_train, X_test, y_train, y_test ):
    classifiers = {
        "customLogistic": CustomlogisticRegression(),
        "LogisiticRegression": LogisticRegression(max_iter=1e4),
        "KNearest": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
        "MLPClassifier": MLPClassifier(),
    }


    f, axes = plt.subplots(1, 5, figsize=(20, 5), sharey='row')

    for i, (key, classifier) in enumerate(classifiers.items()):
        # if classifier == CustomlogisticRegression():
        #     classifier.fit(X_train,y_train)
        #     y_pred = classifier.predict(X_test)
        # else:
        #     y_pred = classifier.fit(X_train, y_train).predict(X_test)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cf_matrix = metrics.confusion_matrix(y_test, y_pred)
        disp =  metrics.ConfusionMatrixDisplay(cf_matrix)
        disp.plot(ax=axes[i], xticks_rotation=45)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        aucScore = metrics.auc(fpr, tpr)

        disp.ax_.set_title(key+":"+"{:.2e}".format(aucScore))
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i != 0:
            disp.ax_.set_ylabel('')

    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    "imBalancedOneHotMinMax"
    "BalancedOneHotMinMax"
    "BalancedCategoricalMinMax"
    f.suptitle("BalancedLabelMinMax")
    f.colorbar(disp.im_, ax=axes)
    plt.show()

def heatmap(data):
    corr = data.corr()
    sns.heatmap(corr,annot = True)
    plt.show()

def main():
    # print("original data")
    bank = pd.read_csv("bank.csv", delimiter=';')
    # print(bank.head())
    # print("after oneHotEncoding")
    # df = oneHotEnc(bank)
    df = labelEncoding(bank)
    # print(df.head())
    # print(df.columns)
    # print(dfOnehot.head())
    # print(dfOnehot.columns)
    # heatmap(df)
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
    multiConfusionPlot(xTrain,xTest,yTrain,yTest)




if __name__ == "__main__":
    main()


