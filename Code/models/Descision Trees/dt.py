import numpy as np
import os

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix

from pymongo import MongoClient

import datetime

import sys
sys.path.append('../..')

import utils.dbUtils
import utils.gensimUtils

client = MongoClient('localhost', 27017)
db = client.TFE
collection = db.results

idx = collection.insert_one({'model' : 'DecisionTreeClassifier', 'date' : datetime.datetime.now(), 'downsampling' : False, 'smote' : False, 'corpus' : 'news_cleaned'})

print('Creating corpus')
corpus = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'type' : {'$in' : ['fake', 'reliable']}})
print('Creating labels')
y = np.array([x for x in corpus.iterTags()])

train_accuracy = []
test_accuracy = []
kf = KFold(n_splits=3, shuffle = True)
i = 1
for i, (train_index, test_index) in enumerate(kf.split(y)):
    print('Train and test set {}'.format(i))
    model = DecisionTreeClassifier()
    vectorizer = TfidfVectorizer()
    print('\t Fiting tf-idf')
    X_train = vectorizer.fit_transform([' '.join(corpus[i]) for i in train_index])
    X_test = vectorizer.transform([' '.join(corpus[i]) for i in test_index])
    y_train = y[train_index]
    y_test = y[test_index]
    print('\t fiting model')
    model.fit(X_train, y_train)
    print('\t Testing model')
    train_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))
    #print("Training accuracy : {}".format(model.score(X_train, y_train)))
    #print("Test accuracy : {}".format(model.score(X_test, y_test)))
    #print("Classification report for test set")
    #print(classification_report(y_test, model.predict(X_test)))
    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)
    collection.update_one({'_id' : idx.inserted_id}, {'$push' : {'classification_report' : crp, 'train_accuracy' : model.score(X_train, y_train), 'test_accuracy' : model.score(X_test, y_test)}})

collection.update_one({'_id' : idx.inserted_id}, {'$set' : {'mean_test_accuracy' : np.mean(test_accuracy) }})



idx = collection.insert_one({'model' : 'DecisionTreeClassifier', 'date' : datetime.datetime.now(), 'downsampling' : True, 'smote' : False, 'corpus' : 'news_cleaned'})

print('Creating corpus')
corpus = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'type' : {'$in' : ['fake', 'reliable']}, 'domain' : {'$nin' : ['nytimes.com', 'beforeitsnews.com']}})
print('Creating labels')
y = np.array([x for x in corpus.iterTags()])

train_accuracy = []
test_accuracy = []
kf = KFold(n_splits=3, shuffle = True)
for i, (train_index, test_index) in enumerate(kf.split(y)):
    print('Train and test set {}'.format(i))
    model = DecisionTreeClassifier()
    vectorizer = TfidfVectorizer()
    print('\t Fiting tf-idf')
    X_train = vectorizer.fit_transform([' '.join(corpus[i]) for i in train_index])
    X_test = vectorizer.transform([' '.join(corpus[i]) for i in test_index])
    y_train = y[train_index]
    y_test = y[test_index]
    print('\t fiting model')
    model.fit(X_train, y_train)
    print('\t Testing model')
    train_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))
    #print("Training accuracy : {}".format(model.score(X_train, y_train)))
    #print("Test accuracy : {}".format(model.score(X_test, y_test)))
    #print("Classification report for test set")
    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)
    collection.update_one({'_id' : idx.inserted_id}, {'$push' : {'classification_report' : crp, 'train_accuracy' : model.score(X_train, y_train), 'test_accuracy' : model.score(X_test, y_test)}})

collection.update_one({'_id' : idx.inserted_id}, {'$set' : {'mean_test_accuracy' : np.mean(test_accuracy) }})