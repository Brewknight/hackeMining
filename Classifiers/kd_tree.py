import numpy as np
import itertools as it
import math

class KDTree:
    # Tree Node
    class __KDNode:
        def __init__(self):
            self.value = -1
            self.left = None
            self.right = None
    # Container for Vectors
    class __Bucket:
        def __init__(self, X):
            self.vectors = X

    def __init__(self, data, k_neighbours, balanced=False):
        if not data:
            print "No Data"
            return

        self.balanced = balanced

        # y is actual dimensions of dataset
        y = len(data[0][0])

        # d is calculated dimensions of dataset
        # d is the least dimensions we need in order to have at least k vectors in each bucket
        d = math.log( len(data) / k_neighbours, 2 )
        d = int(d)

        if d > y :
            self.dimensions = y
        else:
            self.dimensions = d
        
        # -1 here for indexing
        self.dimensions -= 1

        self.root = self.__KDNode() # Initialize root node
        self.root = self.create(self.root, data, 0) # Start creating the tree

    def create(self, Node, data, dimension):
        lefts = list()
        rights = list()
        
        if self.balanced:
            data = sorted(data, key=lambda t: t[0][dimension])
            Node.value = data[int(len(data) / 2)][0][dimension] # Assign median value in current node
            lefts = data[ : int(len(data) / 2) ]
            rights = data[int(len(data) / 2) : ]
        else:
            Node.value = np.median([vec[0] for vec in data])

            for t in data:
                if t[0][dimension] < Node.value:
                    lefts.append(t)
                else:
                    rights.append(t)

        if dimension >= self.dimensions: # If we reached maximum depth, create buckets and return Node
            Node.left = self.__Bucket(lefts)
            Node.right = self.__Bucket(rights)
            return Node

        else: # Otherwise create left and right children-nodes of current node recursively
            Node.left = self.__KDNode()
            Node.right = self.__KDNode()

            Node.left = self.create(Node.left, lefts, dimension + 1)
            Node.right = self.create(Node.right, rights, dimension + 1)
            return Node

    # Typical binary tree recursive search
    def __recsearch(self, node, vector, dimension):
        if dimension > self.dimensions:
            return node.vectors

        if vector[dimension] < node.value:
            return self.__recsearch(node.left, vector, dimension + 1)
        else:
            return self.__recsearch(node.right, vector, dimension + 1)

    def search(self, vector):
        return self.__recsearch(self.root, vector, 0)
        
        