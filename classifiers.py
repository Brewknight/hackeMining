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


original_train_data = pd.read_csv("./datasets/train_set.csv", sep="\t")
#original_train_data = original_train_data[0:1000]

# Classifiers in a dict
Classifiers = dict()
Classifiers['RandomForest'] = RandomForestClassifier()
Classifiers['SupportVector'] = svm.SVC(C=10, kernel='rbf', gamma=0.0001)
Classifiers['MultinomialNB'] = MultinomialNB()
#Classifiers['KNearest'] = KNN(k_neighbours=20, dense=True)

clf = Classifiers['MultinomialNB']
svd = TruncatedSVD(n_components=100)
mms = preprocessing.MinMaxScaler(feature_range=(0, 100))

# Including Title in contents so we can use it to better identify a document
titles = original_train_data['Title']
contents = original_train_data['Content']
contentslist = list()
for tit, con, in it.izip(titles, contents):
    words = con.split()
    length = len(words)
    titwords = tit.split()
    titlength = len(tit)
    for i in xrange((length / titlength) / 10):
        con += tit
    contentslist.append(con)

# Stemming
# ps = PorterStemmer()
# stemmedlist = list()
# for doc in contentslist:
#     doc.decode("utf8")
#     words = word_tokenize(doc)
#     stemmed = ""
#     for word in words:
#         print word
#         stemmed += ps.stem(word)
#         #print stemmed
#     stemmedlist.append(stemmed)
# asdsadfd

le = preprocessing.LabelEncoder()

y = le.fit_transform(original_train_data['Category'])

cv = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X = cv.fit_transform(contentslist)
# Truncating and MinMaxScaling. MinMaxScaling is used for NaiveBayes, since TruncateSVD returns negative values
X = svd.fit_transform(X)
X = mms.fit_transform(X)

ksplits = 10
kf = KFold(n_splits=ksplits, shuffle=False)

for key, clf in Classifiers.iteritems():
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

    avgprec = precs / ksplits
    avgrec = recs / ksplits
    avgf1 = f1s / ksplits
    avgacc = accs / ksplits

    print key
    print "Precision: " + str(avgprec)
    print "Recall: " + str(avgrec)
    print "F1: " + str(avgf1)
    print "Accuracy: " + str(avgacc)
    print "\n"
