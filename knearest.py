import itertools as it
from math import sqrt


class KNearest:
    def __init__(self, k_neighbours=5, dense=False):
        self.k_neighbours = k_neighbours
        self.data = list()
        self.dense = dense
            

    def fit(self, X, y):
        self.label_set = set(y)
        for con, lab in it.izip(X, y):
            self.data.append( (con, lab) )



    def predict(self, X_test):
        predictions = list()
        for u in X_test:
            dists = list()

            for v, lab in self.data:
                dists.append( (self.__distance(u, v), lab) )
            dists = sorted(dists, key=lambda dist: dist[0])

            nearest = dists[:self.k_neighbours]
            predictions.append(self.__findMajority(nearest))
        
        return predictions


    def __findMajority(self, dists):
        labels = dict()
        for l in self.label_set:
            labels[l] = 0
        for dist, lab in dists:
            labels[lab] += 1

        maxval = 0
        maxkey = -1
        for key, value in labels.iteritems():
            if value > maxval:
                maxval = value
                maxkey = key
        return maxkey


    def __distance(self, U, V):
        s = 0
        if not self.dense:
            U = U.toarray()
            U = U[0]
            V = V.toarray()
            V = V[0]
        for xu, xv in it.izip(U, V):
            s += (xu - xv) ** 2
        dist = sqrt(s)
        return dist