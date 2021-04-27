"""
K-means clustering.
"""

import numpy as np
from matplotlib import pyplot as plt


def analyze_kmeans():
    """
    Top-level wrapper to iterate over a bunch of values of k and plot the
    distortions and misclassification rates.
    """
    X = np.genfromtxt("digit.txt")
    y = np.genfromtxt("labels.txt", dtype=int)
    distortions = []
    errs = []
    ks = range(1, 3)
    for k in ks:
        distortion, err = analyze_one_k(X, y, k)
        distortions.append(distortion)
        errs.append(err)
    fig, ax = plt.subplots(2, figsize=(8, 6))

    ax[0].plot(ks, distortions, marker=".")
    ax[0].set_ylabel("Distortion")
    ax[1].plot(ks, errs, marker=".")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Mistake rate")
    ax[0].set_title("k-means performance")
    fig.savefig("kmeans.png")


def analyze_one_k(X, y, k):
    """
    Run the k-means analysis for a single value of k. Return the distortion and
    the mistake rate.
    """
    print ("Running k-means with k={0}".format(k))
    clust = cluster(X, y, k)
    print ("Computing classification error.")
    err = compute_mistake_rate(y, clust)
    return clust["distortion"], err


def cluster(X, y, k, n_starts=5):
    """
    Run k-means a total of n_starts times. Returns the results from the run that
    had the lowest within-group sum of squares (i.e. the lowest distortion).

    Inputs
    ------
    X is an NxD matrix of inputs.
    y is a Dx1 vector of labels.
    n_starts says how many times to randomly re-initialize k-means. You don't
        need to change this.

    Outputs
    -------
    The output is a dictionary with the following fields:
    Mu is a kxD matrix of cluster centroids
    z is an Nx1 vector assigning points to clusters. So, for instance, if z[4] =
        2, then the algorithm has assigned the 4th data point to the second
        cluster.
    distortion is the within-group sum of squares, a number.
    """
    def loop(X, i):
        """
        A single run of clustering.
        """
        Mu = initialize(X, k)
        N = X.shape[0]
        z = np.repeat(-1, N)        # So that initially all assignments change.
        while True:
            old_z = z
            z = assign(X, Mu)       # The vector of assignments z.
            Mu = update(X, z, k)    # Update the centroids
            if np.all(z == old_z):
                distortion = compute_distortion(X, Mu, z)
                return dict(Mu=Mu, z=z, distortion=distortion)

    # Main function body
    print ("Performing clustering.")
    results = [loop(X, i) for i in range(n_starts)]
    # print(results)
    best = min(results, key=lambda entry: entry["distortion"])
    best["digits"] = label_clusters(y, k, best["z"])
    return best


def assign(X, Mu):
    """
    Assign each entry to the closest centroid. Return an Nx1 vector of
    assignments z.
    X is the NxD matrix of inputs.
    Mu is the kxD matrix of cluster centroids.
    """
    # TODO: Compute the assignments z.
    # print(Mu)
    z = []
    for entry in X:
        temK = 0
        temMin = float('inf')
        for ind, k in enumerate(Mu):
            curMin = np.sum((entry - k)**2)
            if curMin < temMin:
                temMin = curMin
                temK = ind
        z.append(temK)
    # print(z)
    # print(len(z))
    return z


def update(X, z, k):
    """
    Update the cluster centroids given the new assignments. Return a kxD matrix
    of cluster centroids Mu.
    X is the NxD inputs as always.
    z is the Nx1 vector of cluster assignments.
    k is the number of clusters.
    """
    # TODO: Compute the cluster centroids Mu.
    Mu = []
    for clu in range(k):
        indexes = [i for i, x in enumerate(z) if x == clu]
        curTotal = []
        for ind in indexes:
            curTotal.append(X[ind])
        tempCurTotal = np.array(curTotal)
        curMean = tempCurTotal.mean(0)
        # print(curMean.shape)
        Mu.append(curMean)
    Mu = np.array(Mu)
    # print(Mu.shape)
    return Mu


def compute_distortion(X, Mu, z):
    """
    Compute the distortion (i.e. within-group sum of squares) implied by NxD
    data X, kxD centroids Mu, and Nx1 assignments z.
    """
    # TODO: Compute the within-group sum of squares (the distortion).
    tempdistortion = 0
    # print(len(Mu))
    for clu in range(len(Mu)):
        indexes = [i for i, x in enumerate(z) if x == clu]
        curDis = 0
        for ind in indexes:
            curDis+= (np.sum((X[ind]  - Mu[clu])**2))**1/2**2
        tempdistortion+=curDis
    # distortion = np.array(tempdistortion)
    print(tempdistortion)
    return tempdistortion


def initialize(X, k):
    """
    Randomly initialize the kxD matrix of cluster centroids Mu. Do this by
    choosing k data points randomly from the data set X.
    """
    # TODO: Initialize Mu.

    random_state = np.random
    temp = random_state.permutation(X)
    Mu = temp[0:k,:]
    # print(Mu)
    # print(k)
    # print(Mu.shape)
    return Mu

def most_common(lst):
    return max(set(lst), key=lst.count)

def label_clusters(y, k, z):
    """
    Label each cluster with the digit that occurs most requently for points
    assigned to that cluster.
    Return a kx1 vector labels with the label for each cluster.
    For instance: if 20 points assigned to cluster 0 have label "3", and 40 have
    label "5", then labels[0] should be 5.

    y is the Nx1 vector of digit labels for the data X
    k is the number of clusters
    z is the Nx1 vector of cluster assignments.
    """
    # TODO: Compute the cluster labelings.
    labels = []
    for clu in range(k):
        indexes = [i for i, x in enumerate(z) if x == clu]
        curTotal = []
        for ind in indexes:
            curTotal.append(y[ind])
        culab = most_common(curTotal)
        labels.append(culab)
    labels = np.array(labels)
    return labels


def compute_mistake_rate(y, clust):
    """
    Compute the mistake rate as discussed in section 3.4 of the homework.
    y is the Nx1 vector of true labels.
    clust is the output of a run of clustering. Two fields are relevant:
    "digits" is a kx1 vector giving the majority label for each cluster
    "z" is an Nx1 vector of final cluster assignments.
    """
    def zero_one_loss(xs, ys):
        return sum(xs != ys) / float(len(xs))

    y_hat = clust["digits"][clust["z"]]
    return zero_one_loss(y, y_hat)


def main():
    analyze_kmeans()


if __name__ == '__main__':
    main()
