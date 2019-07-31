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

from pymongo import MongoClient

import datetime

import sys
sys.path.append('../..')

import utils.dbUtils
import utils.gensimUtils

client = MongoClient('localhost', 27017)
db = client.TFE
collection = db.results

def train_and_test(experiment_id, max_features = None):
    print("Using max features : {}".format(max_features))
    idx = collection.insert_one({'date' : datetime.datetime.now(), 'corpus' : 'news_cleaned', 'max_features' : max_features, 'experiment_id' : experiment_id})

    print("Making dataset")

    train = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'split' : 'train'})
    y_train = np.array([x for x in train.iterTags()])

    test = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'split' : 'valid'})
    y_test = np.array([x for x in test.iterTags()])

    print("Fiting tf-idf")

    vectorizer = TfidfVectorizer(max_features = max_features)		
    X_train = vectorizer.fit_transform([' '.join(news) for news in train])	
    X_test = vectorizer.transform([' '.join(news) for news in test])

    print("Fiting linearSVC")

    model = LinearSVC()
    model.fit(X_train, y_train)

    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)

    collection.update_one({'_id' : idx.inserted_id}, 
        {
        '$push' : 
            {'report' : 
                {'model' : 'LinearSVC', 
                'classification_report' : crp, 
                'train_accuracy' : model.score(X_train, y_train), 
                'test_accuracy' : model.score(X_test, y_test),
                'confusion matrix' : 
                    {
                    'train' : list(map(int, confusion_matrix(y_train, model.predict(X_train), labels=['fake', 'reliable']).ravel())),
                    'test' : list(map(int, confusion_matrix(y_test, model.predict(X_test), labels=['fake', 'reliable']).ravel()))
                    }
                }
            }
        })

    print("MultinomialNB")

    model = MultinomialNB()
    model.fit(X_train, y_train)

    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)

    collection.update_one({'_id' : idx.inserted_id}, 
        {
        '$push' : 
            {'report' : 
                {'model' : 'MultinomialNB', 
                'classification_report' : crp, 
                'train_accuracy' : model.score(X_train, y_train), 
                'test_accuracy' : model.score(X_test, y_test),
                'confusion matrix' : 
                    {
                    'train' : list(map(int, confusion_matrix(y_train, model.predict(X_train), labels=['fake', 'reliable']).ravel())),
                    'test' : list(map(int, confusion_matrix(y_test, model.predict(X_test), labels=['fake', 'reliable']).ravel()))
                    }
                }
            }
        })

    print("DecisionTreeClassifier")

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)

    collection.update_one({'_id' : idx.inserted_id}, 
        {
        '$push' : 
            {'report' : 
                {'model' : 'DecisionTreeClassifier', 
                'classification_report' : crp, 
                'train_accuracy' : model.score(X_train, y_train), 
                'test_accuracy' : model.score(X_test, y_test),
                'confusion matrix' : 
                    {
                    'train' : list(map(int, confusion_matrix(y_train, model.predict(X_train), labels=['fake', 'reliable']).ravel())),
                    'test' : list(map(int, confusion_matrix(y_test, model.predict(X_test), labels=['fake', 'reliable']).ravel()))
                    }
                }
            }
        })
    print("RidgeClassifier")

    model = RidgeClassifier()
    model.fit(X_train, y_train)

    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)

    collection.update_one({'_id' : idx.inserted_id}, 
        {
        '$push' : 
            {'report' : 
                {'model' : 'RidgeClassifier', 
                'classification_report' : crp, 
                'train_accuracy' : model.score(X_train, y_train), 
                'test_accuracy' : model.score(X_test, y_test),
                'confusion matrix' : 
                    {
                    'train' : list(map(int, confusion_matrix(y_train, model.predict(X_train), labels=['fake', 'reliable']).ravel())),
                    'test' : list(map(int, confusion_matrix(y_test, model.predict(X_test), labels=['fake', 'reliable']).ravel()))
                    }
                }
            }
        })


if __name__ == "__main__":
    max_features = [50000, 100000, 250000, 500000, 1000000]
    for features in max_features:
        train_and_test(13, features)