# Data Processing and KFold implementation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from knearest import KNearest as KNN

# Metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# General Utility
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt

original_train_data = pd.read_csv("../datasets/train_set.csv", sep="\t")

clf = KNN(k_neighbours=15, dense=True, balanced=True)
n_components = 2

f = open("../datasets/LSIaccuracy.csv", mode="w+")
f.write("Components\tAccuracy\n")

while(n_components <= 500):
    svd = TruncatedSVD(n_components=n_components)

    le = preprocessing.LabelEncoder()

    y = le.fit_transform(original_train_data['Category'])

    cv = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)

    X = cv.fit_transform(original_train_data['Content'])

    X = svd.fit_transform(X)

    ksplits = 10
    kf = KFold(n_splits=ksplits, shuffle=False)


    accs = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)    
        predictions = clf.predict(X_test)

        accs += accuracy_score(y_test, predictions)

    avgacc = accs / ksplits

    f.write(str(n_components) + "\t")
    f.write(str(avgacc) + "\n")
    print avgacc
    print n_components

    n_components += 5

f.close()
data = pd.read_csv("../datasets/LSIaccuracy.csv", sep="\t")

plt.plot(data['Components'], data['Accuracy'])
plt.show()