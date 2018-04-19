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

from wordcloud import WordCloud
from stopwords import STOPWORDS

original_train_data = pd.read_csv("./datasets/train_set.csv", sep="\t")

#print original_train_data['Content']

titles = original_train_data['Title']
contents = original_train_data['Content']

for tit, con, in it.izip(titles, contents):
    words = con.split()
    length = len(words)
    titwords = tit.split()
    titlength = len(tit)
    for i in xrange((length / titlength) / 10):
        con += tit
    print con
    break
print " JO "
for i in contents:
    print i
    break

print type(contents)

pd.pandas.core.series.Series.to_string