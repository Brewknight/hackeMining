# Data Processing and KFold implementation
from stopwords import STOPWORDS
from preprocess import preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from knearest import KNearest as KNN
from mymethod import mymethod

# Metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# General Utility
import pandas as pd
import itertools as it


original_train_data = pd.read_csv("../datasets/train_set.csv", sep="\t")

# Classifiers in a dict
Classifiers = dict()
Classifiers['MultinomialNB'] = MultinomialNB()
Classifiers['RandomForest'] = RandomForestClassifier()
Classifiers['SupportVector'] = svm.SVC(C=100, kernel='rbf', gamma=0.0001)
Classifiers['KNearest'] = KNN(k_neighbours=15, dense=True, balanced=True)

Scores = dict()
Scores['RandomForest'] = list()
Scores['SupportVector'] = list()
Scores['MultinomialNB'] = list()
Scores['KNearest'] = list()
Scores['MyMethod'] = list()

svd = TruncatedSVD(n_components=100)

X = original_train_data['Content']

le = preprocessing.LabelEncoder()

y = le.fit_transform(original_train_data['Category'])

cv = CountVectorizer(stop_words=STOPWORDS)
X = cv.fit_transform(X)

XMNB = X
Xelse = svd.fit_transform(X)

ksplits = 10
kf = KFold(n_splits=ksplits, shuffle=False)

for key, clf in Classifiers.iteritems():

    if key == 'MultinomialNB':
        Xiter = XMNB
    else:
        Xiter = Xelse

    precs = 0
    recs = 0
    f1s = 0
    accs = 0

    for train_index, test_index in kf.split(Xiter):
        X_train, X_test = Xiter[train_index], Xiter[test_index]
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

    Scores[key].append(avgacc)
    Scores[key].append(avgprec)
    Scores[key].append(avgrec)
    Scores[key].append(avgf1)

Scores['MyMethod'] = mymethod()

f = open("../datasets/EvaluationMetric_10fold.csv", mode="w+")

f.write("Statistic Measure\tNaive Bayes\tRandom Forest\tSVM\tKNN\tMy Method\n")

for i in xrange(4):
    if i == 0:
        f.write("Accuracy")
    if i == 1:
        f.write("Precision")
    if i == 2:
        f.write("Recall")
    if i == 3:
        f.write("F-Measure")
    
    f.write("\t" + str(Scores['MultinomialNB'][i]))
    f.write("\t" + str(Scores['RandomForest'][i]))
    f.write("\t" + str(Scores['SupportVector'][i]))
    f.write("\t" + str(Scores['KNearest'][i]))
    f.write("\t" + str(Scores['MyMethod'][i]))
    f.write("\n")
f.close()