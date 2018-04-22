from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from knearest import KNearest as KNN
from preprocess import preprocess

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
from sklearn.decomposition import TruncatedSVD

original_train_data = pd.read_csv("../datasets/train_set.csv", sep="\t")
train_data = original_train_data[:7000]
topred = original_train_data[7000:10000]

svd = TruncatedSVD(n_components=100)

clf = KNN(k_neighbours=15, dense=True, balanced=True)

le = preprocessing.LabelEncoder()

y = le.fit_transform(train_data['Category'])

train_data = preprocess(train_data)
cv = CountVectorizer()
X = cv.fit_transform(train_data)
X = svd.fit_transform(X)

clf.fit(X, y)    
truepreds = le.transform(topred['Category'])

topred = preprocess(topred)
topred = cv.transform(topred)
topred = svd.transform(topred)

predictions = clf.predict(topred)


print classification_report(truepreds, predictions, target_names=list(le.classes_))