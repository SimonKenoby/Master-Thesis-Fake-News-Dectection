import numpy as np
import os

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA, TruncatedSVD


from pymongo import MongoClient

import datetime
import argparse

import sys
sys.path.append('../..')

import utils.dbUtils
import utils.gensimUtils

client = MongoClient('192.168.178.25', 27017)
db = client.TFE
collection = db.results

def train_test(X_train, X_test, y_train, y_test, max_features = None, experiment = None):
    idx = collection.insert_one({'date' : datetime.datetime.now(), 'experiment_id' : experiment, 'corpus' : 'liar-liar', 'max_features' : max_features})


    vectorizer = TfidfVectorizer(max_features = max_features)		
    X_train = vectorizer.fit_transform(X_train)	
    X_test = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train, y_train)
    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)

    collection.update_one({'_id' : idx.inserted_id}, {'$push' : {'report' : {'model' : 'LinearSVC', 'classification_report' : crp, 'train_accuracy' : model.score(X_train, y_train), 'test_accuracy' : model.score(X_test, y_test)}}})
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)
    collection.update_one({'_id' : idx.inserted_id}, {'$push' : {'report' : {'model' : 'MultinomialNB', 'classification_report' : crp, 'train_accuracy' : model.score(X_train, y_train), 'test_accuracy' : model.score(X_test, y_test)}}})

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)
    collection.update_one({'_id' : idx.inserted_id}, {'$push' : {'report' : {'model' : 'DecisionTreeClassifier', 'classification_report' : crp, 'train_accuracy' : model.score(X_train, y_train), 'test_accuracy' : model.score(X_test, y_test)}}})

    model = RidgeClassifier()
    model.fit(X_train, y_train)
    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)
    collection.update_one({'_id' : idx.inserted_id}, {'$push' : {'report' : {'model' : 'RidgeClassifier', 'classification_report' : crp, 'train_accuracy' : model.score(X_train, y_train), 'test_accuracy' : model.score(X_test, y_test)}}})

if __name__ == "__main__":
    train = utils.dbUtils.TokenizedIterator('liar_liar', filters = {'split' : 'train'})
    y_train = np.array([x for x in train.iterTags()])
    train = [' '.join(news) for news in train]

    test = utils.dbUtils.TokenizedIterator('liar_liar', filters = {'split' : 'valid'})
    y_test = np.array([x for x in test.iterTags()])
    test = [' '.join(news) for news in test]


    for features in max_features:
        train_test(train, test, y_train, y_test, features, experiment = 5)