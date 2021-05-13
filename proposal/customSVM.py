import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import pandas as pd
import math
import seaborn as sns
from scipy import optimize

from sklearn import metrics, svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


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

    im = ax.scatter(X[:, 0], X[:, 22], c=y, cmap=cmap, alpha=0.5, label=label)
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
    # ax.set_xlim(bound[0])
    # ax.set_ylim(bound[1])
    ax.set_xlabel("age")
    ax.set_ylabel("balance")
    cb = plt.colorbar(im, ax=ax)
    loc = np.arange(-1, 1, 1)
    cb.set_ticks(loc)
    cb.set_ticklabels(['-1', '1'])

    plt.show()


def plotHeatMap(X, classes, title=None, fmt='.2g', ax=None, xlabel=None, ylabel=None):
    """ Fix heatmap plot from Seaborn with pyplot 3.1.0, 3.1.1
        https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
    """
    ax = sns.heatmap(X, xticklabels=classes, yticklabels=classes, annot=True,
                     fmt=fmt, cmap=plt.cm.Blues, ax=ax)  # notation: "annot" not "annote"
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    plt.show()


def plotConfusionMatrix(yTrue, yEst, classes, title=None, fmt='.2g', ax=None):
    plotHeatMap(metrics.confusion_matrix(yTrue, yEst), classes, title, fmt, ax, xlabel='Estimations',
                ylabel='True values')

class customizedKernelSVM:
    def __init__(self,C):
        pass

class testNumpyFeature:
    def kernel(self,x1,x2):

        # return (x1*x2)+1
        diff  = x1 - x2
        return np.exp(-np.dot(diff,diff)*len(x1)/2)

    def newAxis(self,X,y):
        # hXX = np.apply_along_axis(lambda x2: self.kernel(x2, x2), 1, X)
        hXX = np.apply_along_axis(lambda x1: np.apply_along_axis(lambda x2: self.kernel(x1, x2), 1, X), 1, X)
        print(len(X[0]))
        print(X)
        print(len(hXX[0]))
        print(hXX)
        # print(type(X))
        # print(X.shape)
        # print(y.shape)
        # print(y[:,np.newaxis])
        # print(y.shape)
        # print(X*y[:,np.newaxis])
class MaxMarginClassifier:
    def __init__(self):
        self.alpha = None
        self.w = None
        self.supportVectors  = None
    def fit(self, X,y):
        N = len(y)
        Xy = X*y[:,np.newaxis]
        GramyXy = np.matmul(Xy, Xy.T)

        def Ld0(G,alpha):
            return alpha.sum() - 0.5 * alpha.dot(alpha.dot(G))

        def Ld0dAlpha(G,alpha):
            return np.ones_like(alpha) - alpha.dot(G)

        A = -np.eye(N)
        b = np.zeros(N)
        constraints = ({'type':'eq','fun':lambda a: np.dot(a,y),'jac':lambda a:y})

class KernelSvm:
    def __init__(self,C ,kernel):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.supportVectors = None

    def fit(self, X,y):
        N = len(y)
        hXX = np.apply_along_axis(lambda x1:np.apply_along_axis(lambda x2: self.kernel(x1,x2),1,X),1,X)

        yp = y.reshape(-1,1)
        GramHXy = hXX* np.matmul(yp,yp.T)

        def Ld0(G,alpha):
            return alpha.sum() - 0.5 * alpha.dot(alpha.dot(G))
        def Ld0Alpha(G,alpha):
            return np.ones_like(alpha) - alpha.dot(G)
        A = np.vstack((-np.eye(N),np.eye(N)))
        b = np.hstack((np.zeros(N), self.C*np.ones(N)))

        constraints = ({'type':'eq', 'fun':lambda a:np.dot(a,y), 'jac': lambda a:y},
                       {'type': 'ineq', 'fun': lambda a: b- np.dot(A, y), 'jac': lambda a: -A},
                       )
        optres = optimize.minimize(fun = lambda a: -Ld0(GramHXy,a),
                                   x0 = np.ones(N),
                                   method = 'SLSQP',
                                   jac = lambda a:-Ld0Alpha(GramHXy,a),
                                   constraints = constraints
                                   )
        self.alpha = optres.x
        epsilon = 1e-8
        supportIndices = self.alpha > epsilon
        self.supportVectors = X[supportIndices]
        self.supportAlphaY = y[supportIndices] * self.alpha[supportIndices]

    def predict(self,X):
        def predict1(x):
            x1 = np.apply_along_axis(lambda s:self.kernel(s,x),1,self.supportVectors)
            x2 = x1 * self.supportAlphaY
            return np.sum(x2)
        d = np.apply_along_axis(predict1, 1,X)
        return 2 *(d>0) -1

def GRBF(x1, x2):
    diff = x1 - x2
    return np.exp(-np.dot(diff,diff)*len(x1)/2)
def EXPON(x1,x2):
    sigma = 10
    return np.exp(-1*np.sum((x1-x2)**2)**0.5/(2*sigma**2))
def LINEAR(x1, x2):
    return (x1*x2)+1


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
    # heat map is used for illustration
    # heatmap(df)
    # print(type(df))
    # print(df.shape)

    train_MF, test_NF = split_into_train_and_test(df, frac_test=0.95, random_state=np.random.RandomState(0))
    xTest = test_NF[:, :-1]
    yTest = test_NF[:, -1]

    xTrain = train_MF[:, :-1]
    yTrain = train_MF[:, -1]

    xTrain2DPlot = train_MF[:, [0,22]]
    # the above is 2d X
    # print(xTrain2DPlot)
    # print(xTrain2DPlot.shape)
    # print(yTrain)
    # print(yTrain.shape)
    #print correlation map
    return xTrain,yTrain,xTest,yTest
def trainAndTest(model,xTrain,yTrain,xTest,yTest):
    # plotSvm(xTrain2DPlot,yTrain)
    model.fit(xTrain, yTrain)

    # fig, axes = plt.subplots(1, 2, figsize=(16, 3))
    # for model,  title in zip([model30,model30], ['Custom linear SVM','Custom linear SVM']):
    #     yEst = model30.predict(xTest)
    #     plotConfusionMatrix(yTest, yEst, colors, "title is nan")

    # fig, ax = plt.subplots(1, figsize=(11, 7))
    # plotSvm(xTrain,yTrain, support=model30.supportVectors, label='Training', ax=ax)
    # yEst = model30.predict(xTest)
    # plotConfusionMatrix(yTest,yEst,colors,"title is nan", ax)

    # print(yTrain)
    # print(yTrain.shape)
    # print(model.predict(xTrain))
    print(accuracy_score(yTest,model.predict(xTest)))




colors = ['blue','red']
cmap = pltcolors.ListedColormap(colors)
nFeatures = 2
N = 100

def main():
    # testObject = testNumpyFeature()
    # xTrain, yTrain = generateBatchBipolar(1000)
    # model30 = KernelSvm(C= 5, kernel=GRBF)
    # model30.fit(xTrain,yTrain)
    # fig, ax = plt.subplots(1, figsize=(11, 7))
    # plotSvm(xTrain,yTrain, support=model30.supportVectors, label='Training', ax=ax)
    # testObject.newAxis(X,y)
    xTrain,yTrain,xTest,yTest = csvToArray()
    model30 = KernelSvm(C= 5, kernel=GRBF)
    model = svm.SVC(kernel='rbf')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    trainAndTest(model30, xTrain,yTrain,xTest,yTest)

if __name__ == "__main__":
    main()

