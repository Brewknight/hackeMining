import pandas as pd
import itertools as it

from sklearn import preprocessing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from sklearn.decomposition import TruncatedSVD

original_train_data = pd.read_csv("./datasets/train_set.csv", sep="\t")

Classifiers = dict()
Classifiers['RandomForest'] = RandomForestClassifier()
Classifiers['SupportVector'] = svm.SVC()
Classifiers['MultinomialNB'] = MultinomialNB()

clf = Classifiers['SupportVector']
svd = TruncatedSVD(n_components=150)

# SVM Param Grid
svm_param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

svm_gscv = GridSearchCV(estimator=Classifiers['SupportVector'], param_grid=svm_param_grid)

le = preprocessing.LabelEncoder()

y = le.fit_transform(original_train_data['Category'])

cv = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X = cv.fit_transform(original_train_data['Content'])
X = svd.fit_transform(X)

svm_gscv.fit(X, y)
print "SVM Best Params for Whole set:"
print svm_gscv.best_params_

