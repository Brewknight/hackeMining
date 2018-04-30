import itertools as it
from math import sqrt
from kd_tree import KDTree
import time

class KNearest:
    def __init__(self, k_neighbours=5, dense=False, balanced=False):
        self.k_neighbours = k_neighbours
        self.dense = dense
        self.balanced = balanced
        self.data = None
            

    def fit(self, X, y):
        self.label_set = set(y)
        data = list()
        for con, lab in it.izip(X, y):
            if not self.dense:
                con = con.toarray()
                con = con[0]
            data.append( (con, lab) )
        # Create a KDTree using the data given and store it
        self.data = KDTree(data, self.k_neighbours, balanced=self.balanced)


    def predict(self, X_test):
        predictions = list()
        if not self.dense:
            X_test = toArray(X_test)
        for u in X_test:
            dists = list()

            # neighbours =  bucket of vectors to compare u with
            neighbours = self.data.search(u)

            # Make a list of distances between u and each neighbour 
            for n in neighbours:
                dists.append( (self.__distance(u, n[0]), n[1]) )

            # Sort the list so we can get the k closest neighbours
            dists = sorted(dists, key=lambda dist: dist[0])

            nearest = dists[:self.k_neighbours]

            # Find Majority
            predictions.append(self.__findMajority(nearest))
        return predictions


    def __findMajority(self, dists):
        labels = dict()
        # Make a dict of occurences of each label in dists
        for l in self.label_set:
            labels[l] = 0
        for __dist, lab in dists:
            labels[lab] += 1

        # Find max label
        maxval = 0
        maxkey = -1
        for key, value in labels.iteritems():
            if value > maxval:
                maxval = value
                maxkey = key
        return maxkey

    # Euclidean distance of 2 vectors
    def __distance(self, U, V):
        s = 0
        for xu, xv in it.izip(U, V):
            s += (xu - xv) ** 2
        dist = sqrt(s)
        return dist

# Convert non dense array to dense array
# This is used to convert mainly sparse matrices
# and other structures used by users.
# They need to have the .toarray() method implemented
def toArray(table):
    new = list()
    for i in table:
        i = i.toarray()
        i = i[0]
        new.append(i)
    return new