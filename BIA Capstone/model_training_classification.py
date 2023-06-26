"""This code will do training of a model for predicting if message is Partisan or not"""
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

# partisan bias
posts = []
for row in df.text:
    posts.append(clean_words(row))
clean_posts = [" ".join(post) for post in posts]

# TF-IDF Vectorizer to convert into numeric data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_posts)
y = pd.get_dummies(df.bias).drop('neutral', axis=1).values.ravel()

# spitting data into testing & training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 6)

# Random forest for evaluation

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)

def plot_cm(model, model_type, X_test, y_test, X_train, y_train, class_names):
    """Takes in a fitted classifier, a string for the the model type, 
       X and y training and test data, and a list of the class names. Returns a 7x5 
       confusion matrix for training and test data, with accuracy, precision, recall 
       and F1 scores plotted next to the matrices."""
    
    y_preds = model.predict(X_test)
    class_names = class_names
    fig, ax = plt.subplots(figsize=(7, 5))
    cm_display_test = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                                           display_labels=class_names)
    cm_display_test.plot(ax=ax, cmap=plt.cm.Reds, xticks_rotation='vertical')
    
    plt.text(x=-.5, y=-.6, s="{} Test Confusion Matrix".format(model_type), fontsize=15, fontname='silom')
    plt.text(x=2.1, y=.1, s="Accuracy: {}".format(float(round(accuracy_score(y_test, y_preds),3))), fontsize=14)
    plt.text(x=2.1, y=.3, s="Precision: {}".format(float(round(precision_score(y_test, y_preds), 3))),fontsize=14)
    plt.text(x=2.1, y=.5, s="Recall: {}".format(float(round(recall_score(y_test, y_preds), 3))),fontsize=14)
    plt.text(x=2.1, y=.7, s="F1: {}".format(float(round(f1_score(y_test, y_preds), 3))),fontsize=14)
    
    plt.savefig("RF_Test_{}.png".format(model_type))
    
    y_preds = model.predict(X_train)
    class_names = class_names
    fig, ax = plt.subplots(figsize=(7, 5))
    cm_display_train = ConfusionMatrixDisplay.from_estimator(model, X_train, y_train,
                                                            display_labels=class_names)
    cm_display_train.plot(ax=ax, cmap=plt.cm.Reds, xticks_rotation='vertical')
    
    plt.text(x=-.5, y=-.6, s="{} Training Confusion Matrix".format(model_type), fontsize=15, fontname='silom')
    plt.text(x=2.1, y=.1, s="Accuracy: {}".format(float(round(accuracy_score(y_train, y_preds),3))), fontsize=14)
    plt.text(x=2.1, y=.3, s="Precision: {}".format(float(round(precision_score(y_train, y_preds), 3))),fontsize=14)
    plt.text(x=2.1, y=.5, s="Recall: {}".format(float(round(recall_score(y_train, y_preds), 3))),fontsize=14)
    plt.text(x=2.1, y=.7, s="F1: {}".format(float(round(f1_score(y_train, y_preds), 3))),fontsize=14)
    
    plt.savefig("RF_Train_{}.png".format(model_type))


plot_cm(rf, 'Baseline RF',
        X_train, y_train, 
        X_test, y_test, 
        class_names=['Neutral', 'Partisan'])

# RandomizedSearchCV for better accuracy
class_weight = ['balanced', 'balanced_subsample']
n_estimators = [50, 100, 150, 500, 1000]
max_features = ['sqrt']
max_depth = [10, 50, 80, 100, 120]
min_samples_split = [2, 5, 6, 7, 10]
min_samples_leaf = [1, 2, 4, 6]
bootstrap = [True, False]
param_grid = {'class_weight' : class_weight,
               'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, 
                               n_iter = 10, cv = 5, random_state=6, n_jobs = -1)
fit = rf_random.fit(X_train,y_train)
best_params = rf_random.best_params_
best_params = best_params
optimized_rf = rf_random.best_estimator_
opt_rf = optimized_rf.fit(X_train,y_train)
plot_cm(optimized_rf, 'Random Search',
        X_train, y_train, 
        X_test, y_test, 
        class_names=['Neutral', 'Partisan'])

# over-sampling minority class with help of Synthetic Minority Oversampling Technique
smote = SMOTE(random_state=6)
X, y = smote.fit_resample(X, y)
counter = Counter(y)
SMOTE_X_train, SMOTE_X_test, SMOTE_y_train, SMOTE_y_test = train_test_split(X, y, test_size=0.2, random_state = 6)
optimized_rf.fit(SMOTE_X_train, SMOTE_y_train)
plot_cm(optimized_rf, 'Smote Random Forest',
        X_train, y_train, 
        X_test, y_test, 
        class_names=['Neutral', 'Partisan'])

# Classifying Message of Posts
X = vectorizer.fit_transform(clean_posts)

le = preprocessing.LabelEncoder()
y = pd.DataFrame(le.fit_transform(df.message)).values.ravel()

# implementing K-Nearest neighbours
X, y = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 9)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


fig, ax = plt.subplots(figsize=(12, 10))
cm = confusion_matrix(y_test, knn.predict(X_test))
cm_normalized = normalize(cm, axis=1, norm='l1')  # Normalize along the rows (axis=1)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_normalized,
                                    display_labels=list(le.classes_))
cm_display.plot(ax=ax, cmap=plt.cm.Reds, xticks_rotation='vertical')
plt.title('KNN Baseline Test Confusion Matrix', fontname='silom', fontsize=17)
plt.savefig("KNN_baseline.png")

n_neighbors = [1, 2, 3, 5, 10, 15]
p=[1,2]

param_grid = {'n_neighbors': n_neighbors, 
              'p': p}

knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, param_grid, cv=4)
#fit the best estimator to the data
optimized_knn = knn_grid.fit(X_train, y_train)

y_train_preds = optimized_knn.predict(X_train)
print(accuracy_score(y_train_preds, y_train))
#test accuracy
y_test_preds = optimized_knn.predict(X_test)
print(accuracy_score(y_test_preds, y_test))

# plotting test matrix
y_pred = optimized_knn.predict(X_test)
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# Create the ConfusionMatrixDisplay object with the normalized matrix
cm_display_test = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=list(le.classes_))
fig, ax = plt.subplots(figsize=(12, 10))
cm_display_test.plot(cmap=plt.cm.Reds, ax=ax, xticks_rotation='vertical')
plt.title('KNN Test Confusion Matrix', fontname='silom', fontsize=17)
plt.savefig("knn_test_matrix.png")

#plotting training matrix
# Obtain the predicted labels
y_pred = optimized_knn.predict(X_train)
# Calculate the confusion matrix
cm = confusion_matrix(y_train, y_pred)
# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# Create the ConfusionMatrixDisplay object with the normalized matrix
cm_display_test = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=list(le.classes_))

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(12, 10))
cm_display_test.plot(cmap=plt.cm.Reds, ax=ax, xticks_rotation='vertical')
plt.title('KNN Train Confusion Matrix', fontname='silom', fontsize=17)
plt.savefig("knn_train_matrix.png")
# passing twitter api keys
auth = tweepy.OAuthHandler('8ENI868nIbNraM7EUqXeErSsy',
                           '2lTZWFCaDReX0hJOcLqBw4iHywKzGVCXM0kBcvhZT8GvBiO09d')
api = tweepy.API(auth)

# fetching tweets
def get_tweets(username): 
    
    """Takes in a Twitter username. Returns up to 100 Tweets after Twitter api is defined as 'api'."""
    
    tweets = api.user_timeline(screen_name=username, count=100, tweet_mode="extended")
    
    tweets_list = []
    
    tweets = [tweet.full_text for tweet in tweets]
    
    for tweet in tweets:
        if not tweet[:1].startswith("@") and 'https' not in tweet:
            tweets_list.append(tweet)
            
    return tweets_list

DT_Tweets = get_tweets('POTUS')

# predicting tweets
def predict_tweets(Tweets):
    
    """Takes in a list of Tweets, cleans them, and uses the optimized_knn and optimized_rf models
       to predict the message and bias of Tweets, respectively. Returns a dataframe with the original 
       Tweets and corresponding predictions."""
    
    clean = []
    
    for tweet in Tweets:
        clean.append(clean_words(tweet))
        
    clean = [" ".join(post) for post in clean]
        
    X = vectorizer.transform(clean)

    message_preds = optimized_knn.predict(X)
    bias_preds = optimized_rf.predict(X)
    
    df = pd.DataFrame({'Tweet': Tweets, 'Partisan Bias': bias_preds, 'Message': message_preds})
    
    return df

DT_df = predict_tweets(DT_Tweets)
print('POTUS Tweets:')
print('')
for i in DT_df[DT_df['Partisan Bias'] == 1].Tweet[0:5]:
    print(i)
    print("-------------------------------------")