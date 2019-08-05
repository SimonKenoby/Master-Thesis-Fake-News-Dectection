import numpy as np
import os

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE


from pymongo import MongoClient

import datetime

import sys
sys.path.append('../..')

import utils.dbUtils
import utils.gensimUtils

def train_test1(X_train, X_test, y_train, y_test, db_idx):

    client = MongoClient('localhost', 27017)
    db = client.TFE
    collection = db.results

    max_depth = [10, 100, 1000, 10000, 100000]
    for depths in max_depth:
        print("Training with max_depth = {}".format(depths))
        model = DecisionTreeClassifier(max_depth = depths)
        model.fit(X_train, y_train)
        crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)
        collection.update_one({'_id' : db_idx.inserted_id}, 
                                {'$push' : 
                                {'report' : 
                                {'classification_report' : crp, 'train_accuracy' : model.score(X_train, y_train), 'test_accuracy' : model.score(X_test, y_test), 'max_depth' : depths}}})
    


if __name__ == "__main__":
    client = MongoClient('localhost', 27017)
    db = client.TFE
    collection = db.results

    print("Creating corpus")
    train = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'split' : 'train'})
    y_train = np.array([x for x in train.iterTags()])

    test = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'split' : 'valid'})
    y_test = np.array([x for x in test.iterTags()])
    idx = collection.insert_one({'model' : 'DecisionTreeClassifier', 
        'date' : datetime.datetime.now(), 
        'corpus' : 'news_cleaned', 
        'experiment_id' : 19,
        'SMOTE' : True,
        'description' : 'Testing DecisionTreeClassifier with multiple parameters on fake corpus with train and validation set and SMOTE'})


    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform([' '.join(text) for text in train])
    X_test = vectorizer.transform([' '.join(text) for text in test])

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print(X_train.shape)
    print(X_test.shape)
    print(len(y_train))
    print(len(y_test))

    train_test1(X_res, X_test, y_res, y_test, idx)





