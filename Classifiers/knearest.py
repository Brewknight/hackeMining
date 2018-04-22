import itertools as it
from math import sqrt
from kd_tree import KDTree
import time

class KNearest:
    def __init__(self, k_neighbours=5, dense=False, balanced=False):
        self.k_neighbours = k_neighbours
        self.dense = dense
        self.data = None
            

    def fit(self, X, y):
        self.label_set = set(y)
        data = list()
        for con, lab in it.izip(X, y):
            if not self.dense:
                con = con.toarray()
                con = con[0]
            data.append( (con, lab) )
        self.data = KDTree(data, self.k_neighbours)
        i =1


    def predict(self, X_test):
        start = time.time()
        predictions = list()
        if not self.dense:
            X_test = toArray(X_test)
        print "new"
        for u in X_test:
            start = time.time()
            dists = list()

            neighbours = self.data.search(u)

            for n in neighbours:
                dists.append( (self.__distance(u, n[0]), n[1]) )

            dists = sorted(dists, key=lambda dist: dist[0])

            nearest = dists[:self.k_neighbours]
            predictions.append(self.__findMajority(nearest))
            end = time.time()
            dur = end - start
            print "predict : " + str(dur)
        return predictions


    def __findMajority(self, dists):
        labels = dict()
        for l in self.label_set:
            labels[l] = 0
        for __dist, lab in dists:
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
        for xu, xv in it.izip(U, V):
            s += (xu - xv) ** 2
        dist = sqrt(s)
        return dist

def toArray(table):
    new = list()
    for i in table:
        i = i.toarray()
        i = i[0]
        new.append(i)
    return new