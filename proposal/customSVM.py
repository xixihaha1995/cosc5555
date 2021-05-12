import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors


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
        print(X)
        print(y)
        print(y[:,np.newaxis])
        print(X*y[:,np.newaxis])

def generateBatchBipolar(n,mu = 0.5, sigma = 0.2):
    X = np.random.normal(mu,sigma,(n,2))
    yB = np.random.uniform(0,1,n) > 0.5
    y = 2. * yB-1
    X *= y[:,np.newaxis]
    X -= X.mean(axis= 0)
    return X, y

colors = ['blue','red']
cmap = pltcolors.ListedColormap(colors)
nFeatures = 2
N = 100

def main():
    testObject = testNumpyFeature()
    X, y = generateBatchBipolar(10)
    testObject.newAxis(X,y)

if __name__ == "__main__":
    main()

