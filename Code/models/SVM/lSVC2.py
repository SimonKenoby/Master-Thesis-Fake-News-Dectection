import numpy as np
import os

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
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

    Cs = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 5]
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
    train = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'split' : 'train'})
    y_train = np.array([x for x in train.iterTags()])

    test = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'split' : 'valid'})
    y_test = np.array([x for x in test.iterTags()])

    idx = collection.insert_one({'model' : 'LinearSVC', 
        'date' : datetime.datetime.now(), 
        'smote' : True, 
        'corpus' : 'news_cleaned', 
        'penality' : 'l2', 
        'experiment_id' : 20,
        'description' : 'Testing linearSVC with multiple parameters on news_cleaned with train and validation set and SMOTE'})


    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform([' '.join(text) for text in train])
    X_test = vectorizer.transform([' '.join(text) for text in test])

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    train_test1(X_res, X_test, y_res, y_test, idx)





