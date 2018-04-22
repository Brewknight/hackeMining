
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import itertools as it

from stopwords import STOPWORDS
import time

# Title inclusion and stemming for now
def preprocess(data, titling=True, stemming=True, stopwords=True):
    
    start = time.time()
    
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


    if stemming: 
        ps = PorterStemmer()
        stemmedlist = list()
        for doc in data:
            words = word_tokenize(doc.decode("utf-8"))

            # Stem
            stemmed = ""
            for word in words:   
                # Exclude Stopwords
                if stopwords:          
                    if word in STOPWORDS:
                        continue
                stemmed += (ps.stem(word) + " ")
            stemmedlist.append(stemmed)
        data = stemmedlist

    # Stop Words
    if stopwords and not stemming:
        nostoplist = list()
        for doc in data:
            words = word_tokenize(doc.decode("utf-8"))

            # Exclude stopwords
            nostop = ""
            for word in words:
                if word in STOPWORDS:
                    continue
                nostop += (word +  " ")
            nostoplist.append(nostop)
        data = nostoplist



    end = time.time()

    duration = end - start
    print "Preprocessing time : " + str(duration)
    return data
