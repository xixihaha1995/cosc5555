import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import pandas as pd
import math
import seaborn as sns


def plotLine(ax, xRange, w, x0, label, color='grey', linestyle='-', alpha=1.):
    """ Plot a (separating) line given the normal vector (weights) and point of intercept """
    if type(x0) == int or type(x0) == float or type(x0) == np.float64:
        x0 = [0, -x0 / w[1]]
    yy = -(w[0] / w[1]) * (xRange - x0[0]) + x0[1]
    ax.plot(xRange, yy, color=color, label=label, linestyle=linestyle)


def plotSvm(X, y, support=None, w=None, intercept=0., label='Data', separatorLabel='Separator',
            ax=None, bound=[[-1., 1.], [-1., 1.]]):
    """ Plot the SVM separation, and margin """
    if ax is None:
        fig, ax = plt.subplots(1)

    im = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.5, label=label)
    if support is not None:
        ax.scatter(support[:, 0], support[:, 1], label='Support', s=80, facecolors='none',
                   edgecolors='y', color='y')
        print("Number of support vectors = %d" % (len(support)))
    if w is not None:
        xx = np.array(bound[0])
        plotLine(ax, xx, w, intercept, separatorLabel)
        # Plot margin
        if support is not None:
            signedDist = np.matmul(support, w)
            margin = np.max(signedDist) - np.min(signedDist) * np.sqrt(np.dot(w, w))
            supportMaxNeg = support[np.argmin(signedDist)]
            plotLine(ax, xx, w, supportMaxNeg, 'Margin -', linestyle='-.', alpha=0.8)
            supportMaxPos = support[np.argmax(signedDist)]
            plotLine(ax, xx, w, supportMaxPos, 'Margin +', linestyle='--', alpha=0.8)
            ax.set_title('Margin = %.3f' % (margin))
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlim(bound[0])
    ax.set_ylim(bound[1])
    cb = plt.colorbar(im, ax=ax)
    loc = np.arange(-1, 1, 1)
    cb.set_ticks(loc)
    cb.set_ticklabels(['-1', '1'])

class customizedKernelSVM:
    def __init__(self,C):
        pass

class testNumpyFeature:
    def newAxis(self,X,y):
        print(type(X))
        print(X)
        print(y)
        print(y[:,np.newaxis])
        print(X*y[:,np.newaxis])
class MaxMarginClassifier:
    def __init__(self):
        self.alpha = None
        self.w = None
        self.supportVectors  = None

def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):

    if random_state is None:
        random_state = np.random

    exam,fea = x_all_LF.shape
    N = math.ceil(frac_test*exam)

    # np.random.RandomState(random_state)
    temp = random_state.permutation(x_all_LF)
    x_test_NF = temp[0:N,:]
    x_train_MF = temp[N:,:]
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
    return s

def generateBatchBipolar(n,mu = 0.5, sigma = 0.2):
    X = np.random.normal(mu,sigma,(n,2))
    yB = np.random.uniform(0,1,n) > 0.5
    y = 2. * yB-1
    X *= y[:,np.newaxis]
    X -= X.mean(axis= 0)
    return X, y

def heatmap(data):
    # sns.set_theme()
    # ax = sns.heatmap(data)
    corr = data.corr()
    # corr.style.background_gradient(cmap = 'coolwarm').set_precision(2)
    # plt.matshow(corr)
    sns.heatmap(corr,annot = True)
    # sns.savefig("linear-kernel")
    plt.show()

def csvToArray():
    bank = pd.read_csv("bank.csv", delimiter=';')
    # print(bank.head())
    # print("after oneHotEncoding")
    df = oneHotEnc(bank)

    df.rename(columns={0: 'y'}, inplace=True)
    # print(type(df))
    # print(df.head())

    train_MF, test_NF = split_into_train_and_test(df, frac_test=0.3, random_state=np.random.RandomState(0))
    xTest = test_NF[:, :-1]
    yTest = test_NF[:, -1]

    # print("xTest shape")
    # print(len(xTest))
    # print(len(xTest[0]))
    # print(np.count_nonzero(yTest))
    xTrain = train_MF[:, :-1]
    yTrain = train_MF[:, -1]
    #print correlation map


colors = ['blue','red']
cmap = pltcolors.ListedColormap(colors)
nFeatures = 2
N = 100

def main():
    testObject = testNumpyFeature()
    X, y = generateBatchBipolar(10)
    # testObject.newAxis(X,y)
    csvToArray()

if __name__ == "__main__":
    main()

