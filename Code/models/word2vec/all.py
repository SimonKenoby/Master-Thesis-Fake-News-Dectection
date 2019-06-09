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

def train_and_test(experiment):
    idx = collection.insert_one({'date' : datetime.datetime.now(), 'experiment_number' : experiment, 'downsampling' : True, 'smote' : False, 'corpus' : 'news_cleaned', 'method' : 'word2vec_300'})

    print("Making dataset")

    train = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'type' : {'$in' : ['fake', 'reliable']}, 'domain' : {'$nin' : ['nytimes.com', 'beforeitsnews.com']}})
    y_train = np.array([x for x in train.iterTags()])

    test = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'type' : {'$in' : ['fake', 'reliable']}, 'domain' : {'$in' : ['nytimes.com', 'beforeitsnews.com']}})
    y_test = np.array([x for x in test.iterTags()])

    print("Fiting word2vec")

    vectorizer = w2vVectorizer()
    X_train = np.zeros((len(train), 300))	
    X_test  = np.zeros((len(test), 300))
    for i, news in enumerate(train):
        X_train[i] = vectorizer.transform(news)
    for i, news in enumerate(test):
        X_test[i] = vectorizer.transform(news)
    #X_train = np.array([vectorizer.transform(news) for news in train if news != None])
    #X_test = np.array([vectorizer.transform(news) for news in test if news != None])

    print(X_train.shape)

    print("Fiting linearSVC")

    model = LinearSVC()
    model.fit(X_train, y_train)

    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)

    collection.update_one({'_id' : idx.inserted_id}, {'$push' : {'report' : {'model' : 'LinearSVC', 'classification_report' : crp, 'train_accuracy' : model.score(X_train, y_train), 'test_accuracy' : model.score(X_test, y_test)}}})

    print("DecisionTreeClassifier")

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)

    collection.update_one({'_id' : idx.inserted_id}, {'$push' : {'report' : {'model' : 'DecisionTreeClassifier', 'classification_report' : crp, 'train_accuracy' : model.score(X_train, y_train), 'test_accuracy' : model.score(X_test, y_test)}}})

    print("RidgeClassifier")

    model = RidgeClassifier()
    model.fit(X_train, y_train)

    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)

    collection.update_one({'_id' : idx.inserted_id}, {'$push' : {'report' : {'model' : 'RidgeClassifier', 'classification_report' : crp, 'train_accuracy' : model.score(X_train, y_train), 'test_accuracy' : model.score(X_test, y_test)}}})



if __name__ == "__main__":
    train_and_test(1)