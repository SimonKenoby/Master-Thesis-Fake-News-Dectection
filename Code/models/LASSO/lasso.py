import numpy as np
import os

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

import pandas as pd
import json

from pymongo import MongoClient

import datetime

print('Creating corpus')
corpus = pd.read_json('train.json', lines=True)
y = corpus['type'].values

train_accuracy = []
test_accuracy = []
kf = KFold(n_splits=3, shuffle = True)
for i, (train_index, test_index) in enumerate(kf.split(y)):
    print('Train and test set {}'.format(i))
    model = Lasso()
    vectorizer = TfidfVectorizer()
    print('\t Fiting tf-idf')
    X_train = vectorizer.fit_transform([' '.join(corpus['tokenized_text'].values[i]) for i in train_index])
    X_test = vectorizer.transform([' '.join(corpus['tokenized_text'].values[i]) for i in test_index])
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

    with open('report-{}.json'.format(i), 'w') as outfile:  
        json.dump(crp, outfile)

print("Fitting full model")
model = Lasso()
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform([' '.join(text) for text in corpus['tokenized_text']])
model.fit(X_train, y)

dump(model, 'lasso.joblib') 