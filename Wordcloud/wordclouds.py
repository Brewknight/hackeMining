import pandas as pd
import itertools as it
from wordcloud import WordCloud
from stopwords import STOPWORDS
import os

# if not os.path.exists("./clouds"):
#     os.makedirs("./clouds")

train_data = pd.read_csv("../datasets/train_set.csv", sep="\t")

cm = dict()
Categories = set(train_data['Category'])
for cat in Categories:
    cm[cat] = list()

for con, cat in it.izip(train_data['Content'],  train_data['Category']):
    cm[cat].append(con)


for key, content_list in cm.iteritems():
    temp = ""
    for content in content_list:
        temp += content
    wordcloud = WordCloud(stopwords=STOPWORDS).generate(temp)
    image = wordcloud.to_image()
    
    fp = open("./clouds/" + key + "Cloud.png", "w+")

    image.save(fp)

