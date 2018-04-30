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
        
        # Assign median value in current node
        # and divide data to left and right child
        if self.balanced:
            data = sorted(data, key=lambda t: t[0][dimension])
            Node.value = data[int(len(data) / 2)][0][dimension] 
            lefts = data[ : int(len(data) / 2) ]
            rights = data[int(len(data) / 2) : ]
        else:
            Node.value = np.median([vec[0] for vec in data])

            for t in data:
                if t[0][dimension] < Node.value:
                    lefts.append(t)
                else:
                    rights.append(t)

        # If left or right do not exist, don't create the node.
        # Searching will stop there.
        if not lefts:
            Node.left = None
        if not rights:
            Node.right = None

        if dimension >= self.dimensions: # If we reached maximum depth, create buckets and return Node
            if lefts:
                Node.left = self.__Bucket(lefts)
            if rights:
                Node.right = self.__Bucket(rights)
            return Node

        else: # Otherwise create left and right children-nodes of current node recursively
            if lefts:
                Node.left = self.__KDNode()
                Node.left = self.create(Node.left, lefts, dimension + 1)
            if rights:
                Node.right = self.__KDNode()
                Node.right = self.create(Node.right, rights, dimension + 1)
            return Node

    # Typical binary tree recursive search
    def __recsearch(self, node, vector, dimension):
        # Returns whole bucket
        if dimension > self.dimensions:
            return node.vectors

        # If one node does not eixst, search the other node. This guarantees there will neighbours to search
        if vector[dimension] < node.value:
            if node.left:
                return self.__recsearch(node.left, vector, dimension + 1)
            else: 
                return self.__recsearch(node.right, vector, dimension + 1)
        else:
            if node.right:
                return self.__recsearch(node.right, vector, dimension + 1)
            else:
                return self.__recsearch(node.left, vector, dimension + 1)

    def search(self, vector):
        return self.__recsearch(self.root, vector, 0)
        
        