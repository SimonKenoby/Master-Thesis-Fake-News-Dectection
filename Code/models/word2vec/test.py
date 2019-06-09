import numpy as np
import os

import sklearn
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from w2vVectorizer import w2vVectorizer

from pymongo import MongoClient

import datetime

import sys
sys.path.append('../..')

import utils.dbUtils
import utils.gensimUtils

client = MongoClient('192.168.178.25', 27017)
db = client.TFE
collection = db.results


train = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'type' : {'$in' : ['fake', 'reliable']}, 'domain' : {'$nin' : ['nytimes.com', 'beforeitsnews.com']}}, limit = 2)
y_train = np.array([x for x in train.iterTags()])

test = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'type' : {'$in' : ['fake', 'reliable']}, 'domain' : {'$in' : ['nytimes.com', 'beforeitsnews.com']}}, limit=2)
y_test = np.array([x for x in test.iterTags()])

vectorizer = w2vVectorizer()
X_train = np.zeros((len(train), 300))	
X_test  = np.zeros((len(test), 300))
for i, news in enumerate(train):
    X_train[i] = vectorizer.transform(news)
for i, news in enumerate(test):
    X_test[i] = vectorizer.transform(news)

print(X_train)
print(X_test)
