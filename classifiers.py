# Data Processing and KFold implementation
from stopwords import STOPWORDS
from preprocess import preprocess
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
import time


original_train_data = pd.read_csv("./datasets/train_set.csv", sep="\t")
#original_train_data = original_train_data[0:50]

# Classifiers in a dict
Classifiers = dict()
Classifiers['RandomForest'] = RandomForestClassifier()
Classifiers['SupportVector'] = svm.SVC(C=100, kernel='rbf', gamma=0.0001)
Classifiers['MultinomialNB'] = MultinomialNB()
#Classifiers['KNearest'] = KNN(k_neighbours=20, dense=True)

#clf = Classifiers['MultinomialNB']
svd = TruncatedSVD(n_components=100)
mms = preprocessing.MinMaxScaler(feature_range=(0, 100))

X = preprocess(original_train_data)

le = preprocessing.LabelEncoder()

y = le.fit_transform(original_train_data['Category'])

cv = CountVectorizer()
X = cv.fit_transform(X)

# Truncating and MinMaxScaling. MinMaxScaling is used for NaiveBayes, since TruncateSVD returns negative values
XMNB = X
Xelse = svd.fit_transform(X)

ksplits = 10
kf = KFold(n_splits=ksplits, shuffle=False)



for key, clf in Classifiers.iteritems():
    start = time.time()

    if key == 'MultinomialNB':
        Xiter = XMNB
    else:
        Xiter = Xelse

    precs = 0
    recs = 0
    f1s = 0
    accs = 0
    for train_index, test_index in kf.split(Xiter):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)    
        predictions = clf.predict(X_test)

        precs += precision_score(y_test, predictions, average='micro')
        recs += recall_score(y_test, predictions, average='micro')
        f1s += f1_score(y_test, predictions, average='micro')
        accs += accuracy_score(y_test, predictions)

    avgprec = precs / ksplits
    avgrec = recs / ksplits
    avgf1 = f1s / ksplits
    avgacc = accs / ksplits

    end = time.time()
    duration = end - start
    print key
    print "Precision: " + str(avgprec)
    print "Recall: " + str(avgrec)
    print "F1: " + str(avgf1)
    print "Accuracy: " + str(avgacc)
    print str(key) + " time : " + str(duration)
    print "\n"
