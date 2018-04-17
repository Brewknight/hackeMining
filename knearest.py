import itertools as it
from math import sqrt
from sklearn.decomposition import TruncatedSVD

class KNearest:
    def __init__(self, k_neighbours=5, components=2):
        self.k_neighbours = k_neighbours
        self.data = list()
        self.svd = None
        self.components=components
            

    def fit(self, X, y):
        self.label_set = set(y)
        self.svd = TruncatedSVD(n_components=self.components)
        X = self.svd.fit_transform(X)
            
        for con, lab in it.izip(X, y):
            self.data.append( (con, lab) )


    def predict(self, X_test):
        predictions = list() 
        X_test = self.svd.fit_transform(X_test)
        for u in X_test:
            dists = list()

            for v, lab in self.data:
                dists.append( (distance(u, v), lab) )
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


def distance(U, V):
    s = 0
    for xu, xv in it.izip(U, V):
        s += (xu - xv) ** 2
    dist = sqrt(s)
    return dist