
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import itertools as it

from stopwords import STOPWORDS


# Title inclusion and stemming for now
def preprocess(data, titling=True, stemming=True, stopwords=True):
    
    # Including Title in contents so we can use it to better identify a document
    if titling:
        titles = data['Title']
        contents = data['Content']
        titledList = list()
        for tit, con, in it.izip(titles, contents):
            words = con.split()
            length = len(words)
            titwords = tit.split()
            titlength = len(titwords)
            for i in xrange((length / titlength) / 10):
                con += (" " + tit)
            titledList.append(con)
        data = titledList

    # Stop Words
    if stopwords:
        allstop = list()
        for doc in data:
            words = word_tokenize(doc.decode("utf-8"))

            # Exclude stopwords
            nostoplist = list()
            nostop = ""
            for word in words:
                if word in STOPWORDS:
                    continue
                nostoplist.append(word)
                nostop += (word +  " ")
            allstop.append(nostop)
        data = allstop

    # Stemming
    if stemming:
        ps = PorterStemmer()
        stemmedlist = list()
        for doc in data:
            words = word_tokenize(doc)

            # Stem words
            stemmed = ""
            for word in nostop:
                stemmed += (ps.stem(word) + " ")
            stemmedlist.append(stemmed)
        data = stemmedlist

    return data
