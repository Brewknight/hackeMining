from stopwords import STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import pandas as pd
import itertools as it
#import classifier as clf

train_data = pd.read_csv("./datasets/train_set.csv", sep="\t")
test_data = pd.read_csv("./datasets/test_set.csv", sep="\t")

le = preprocessing.LabelEncoder()

y = le.fit_transform(train_data['Category'])

cv = CountVectorizer(stop_words=STOPWORDS)
X = cv.fit_transform(train_data['Content'])


TestVector = cv.transform(test_data)
clf.fit(X, y)
clf.predict(TestVector)

with open("./datasets/testSet_categories.csv", mode="w+") as f:
#     f.write("Id\tCategory\n")
# Results = 
    