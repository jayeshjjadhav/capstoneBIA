import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from collections import Counter
import tweepy
import re
import string
import imblearn
import joblib

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.collocations import *
from nltk import FreqDist
from nltk.probability import FreqDist

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import normalize


df = pd.read_csv('politic - political_social_media.csv', encoding='utf-8')
# different plots
df['source'].value_counts().sort_values().plot(kind='barh')
plt.title("Source")
plt.show()
df.message.value_counts().sort_values().plot(kind = 'barh')
plt.title("Message")
plt.show()
df.bias.value_counts().sort_values().plot(kind = 'barh')
plt.title("Bias")
plt.show()

# plots for each category in message column
for message in df.message.unique():
    bias = df.bias[df.message==message]
    bias.value_counts().plot(kind = 'barh', title="{} message bias".format(message))
    plt.show()

stopwords_list = [stopwords.words('english')]
for w in ['http ', 'com', 'amp']:
    stopwords_list.append(w)

def clean_words(text):
    
    """Takes in a string or iterable of strings, and returns a cleaned list of words in the text that was
       passed in."""
    #join all strings in text and separate them by white space
    joined_text = "".join(text) 
    #remove all characters that are not letters
    joined_text = re.sub('[^A-Za-z ]+', ' ', joined_text)
    #convert all words in list to their base form, or 'lemma'
    words = [WordNetLemmatizer().lemmatize(word) for word in joined_text]
    #create a list of individual strings for each word in the text 
    words = word_tokenize(joined_text)
    clean_words_list = []
    for word in words:
        #exclude words that don't contribute to the meaning of the text
        stopwords_list = stopwords.words('english')
        for w in ['http', 'com', 'amp', 'www']:
            stopwords_list.append(w)
        if len(word) > 2 and word not in stopwords_list:
            #populate clean words list with remaining words
            clean_words_list.append(word.lower())
    return clean_words_list

def wordcloud_plot(words, type):
    """Takes in a clean list of words and a name for the word cloud. Returns a 25x25 word cloud 
    with the top 1000 most frequent words."""
    clean_value = clean_words(words)
    wc = WordCloud(background_color="Black", max_words=100, max_font_size = 50)
    clean_string = ','.join(clean_value)
    wc.generate(clean_string)
    f = plt.figure(figsize=(30,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title('{} Word Cloud'.format(type), size=30, fontname='Silom')
    plt.show()

neutral_posts = df.text[df.bias == 'neutral']
partisan_posts = df.text[df.bias == 'partisan']
attack_posts = df.text[df.message == 'attack']
policy_posts = df.text[df.message == 'policy']
wordcloud_plot(policy_posts, "Policy")
