# Data Processing and KFold implementation
from stopwords import STOPWORDS
from preprocess import preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import KFold

# Classifier
from sklearn.naive_bayes import MultinomialNB

# Metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# General Utility
import pandas as pd
import time


original_train_data = pd.read_csv("../datasets/train_set.csv", sep="\t")

clf = MultinomialNB()

X = preprocess(original_train_data)

le = preprocessing.LabelEncoder()
y = le.fit_transform(original_train_data['Category'])

cv = CountVectorizer(stop_words=STOPWORDS)
X = cv.fit_transform(X)

ksplits = 10
kf = KFold(n_splits=ksplits, shuffle=False)

start = time.time()

precs = 0
recs = 0
f1s = 0
accs = 0

for train_index, test_index in kf.split(X):
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
print "MultinomialNB"
print "Precision: " + str(avgprec)
print "Recall: " + str(avgrec)
print "F1: " + str(avgf1)
print "Accuracy: " + str(avgacc)
print "MultinomialNB time : " + str(duration)
print "\n"