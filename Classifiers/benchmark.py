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
from itertools import izip

train_data = pd.read_csv("../datasets/train_set.csv", sep="\t")
test_data = pd.read_csv("../datasets/test_set.csv", sep="\t")

clf = MultinomialNB()

X_train = preprocess(train_data)
X_test = preprocess(test_data)

le = preprocessing.LabelEncoder()

y_train = le.fit_transform(train_data['Category'])

cv = CountVectorizer(stop_words=STOPWORDS)
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

start = time.time()

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

end = time.time()

print "time : " + str(end - start)

f = open("../datasets/testSet_categories.csv", mode="w+")

f.write("Id,Category\n")

for ID, Cat in izip(test_data['Id'], le.inverse_transform(predictions)):
    f.write(str(ID) + "," + Cat + "\n")
f.close

    