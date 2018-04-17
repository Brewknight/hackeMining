from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn import preprocessing
import pandas as pd
import itertools as it
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

original_train_data = pd.read_csv("./datasets/train_set.csv", sep="\t")

#print len(train_data)
Classifiers = dict()
Classifiers['RandomForest'] = RandomForestClassifier()
Classifiers['SupportVector'] = svm.SVC(kernel='linear')
Classifiers['MultinomialNB'] = MultinomialNB()

clf = Classifiers['SupportVector']

#data = original_train_data[0:1000]

# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]

#gscv = GridSearchCV(estimator=Classifiers['SupportVector'], param_grid=param_grid)

le = preprocessing.LabelEncoder()

y = le.fit_transform(original_train_data['Category'])

cv = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X = cv.fit_transform(original_train_data['Content'])

#gscv.fit(X, y)
#print gscv.cv_results_.items()
#print gscv.best_params_

kf = KFold(n_splits=10, shuffle=False)

#for key, clf in Classifiers.iteritems():
precs = 0
recs = 0
f1s = 0
accs = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    clf.fit(X_train, y_train)    
    predictions = clf.predict(X_test)

    #print classification_report(y_test, predictions, target_names=list(le.classes_))
    precs += precision_score(y_test, predictions, average='micro')
    recs += recall_score(y_test, predictions, average='micro')
    f1s += f1_score(y_test, predictions, average='micro')
    accs += accuracy_score(y_test, predictions)

avgprec = precs / 10
avgrec = recs / 10
avgf1 = f1s / 10
avgacc = accs / 10

print "RandomForest"
print "Precision: " + str(avgprec)
print "Recall: " + str(avgrec)
print "F1: " + str(avgf1)
print "Accuracy: " + str(avgacc)
print "\n"
