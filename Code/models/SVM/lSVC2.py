import numpy as np
import os

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB


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

    Cs = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 5]
    for C in Cs:
        print("Training with C = {}".format(C))
        model = LinearSVC(C = C)
        model.fit(X_train, y_train)
        crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)
        collection.update_one({'_id' : db_idx.inserted_id}, 
                                {'$push' : 
                                {'report' : 
                                {'classification_report' : crp, 'train_accuracy' : model.score(X_train, y_train), 'test_accuracy' : model.score(X_test, y_test), 'C' : C}}})
    


if __name__ == "__main__":
    client = MongoClient('localhost', 27017)
    db = client.TFE
    collection = db.results

    print("Creating corpus")
    corpus = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'type' : {'$in' : ['fake', 'reliable']}})
    y = np.array([x for x in corpus.iterTags()])

    idx = collection.insert_one({'model' : 'linear_svc', 'date' : datetime.datetime.now(), 'downsampling' : False, 'smote' : False, 'corpus' : 'news_cleaned', 'penality' : 'l2'})


    train_accuracy = []
    test_accuracy = []
    kf = KFold(n_splits=3, shuffle = True)
    for i, (train_index, test_index) in enumerate(kf.split(y)):
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform([' '.join(corpus[i]) for i in train_index])
        X_test = vectorizer.transform([' '.join(corpus[i]) for i in test_index])
        y_train = y[train_index]
        y_test = y[test_index]
        train_test1(X_train, X_test, y_train, y_test, idx)