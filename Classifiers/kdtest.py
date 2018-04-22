from kd_tree import KDTree
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import itertools as it

original_train_data = pd.read_csv("./datasets/train_set.csv", sep="\t")
original_train_data = original_train_data[0:10]

svd = TruncatedSVD(n_components=2)

le = preprocessing.LabelEncoder()

y = le.fit_transform(original_train_data['Category'])

cv = CountVectorizer()
X = cv.fit_transform(original_train_data['Content'])

X = svd.fit_transform(X)

data = list()
for con, lab in it.izip(X, y):
    data.append( (con, lab) )

tree = KDTree(data)

print tree.search([1, 2])


