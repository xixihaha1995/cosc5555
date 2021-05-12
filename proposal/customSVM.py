import numpy as np

def generateBatchBipolar(n,mu = 0.5, sigma = 0.2):
    X = np.random.normal(mu,sigma,(n,2))
    yB = np.random.uniform(0,1,n) > 0.5
    y = 2. * yB-1
    X *= y[:,np.newaxis]
    X -= X.mean(axis= 0)
    return X, y

class customizedKernelSVM:
    def __init__(self,C):
        pass

class testNumpyFeature:
    def newAxis(self,X,y):
        print(X)
        print(y)
        print(y[:,np.newaxis])
        print(X*y[:,np.newaxis])

def main():
    testObject = testNumpyFeature()
    X, y = generateBatchBipolar(10)
    testObject.newAxis(X,y)

if __name__ == "__main__":
    main()

