import numpy as np
import os
import json
import pandas as pd

from  imblearn.over_sampling import SMOTE

import sys
sys.path.append('../..')

import utils.dbUtils
import utils.gensimUtils

import altair as alt

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix


print("Loading data...")
corpus = utils.dbUtils.TokenizedIterator('news_copy', filters = {'type' : {'$in' : ['fake', 'reliable']}, 'domain' : {'$nin' : ['nytimes.com', 'beforeitsnews.com']}})

y = np.array([tag for tag in corpus.iterTags()])
print(np.unique(y, return_counts = True))

train_accuracy = []
test_accuracy = []
print("Starting kfolds tests...")
kf = KFold(n_splits=3, shuffle = True)
for train_index, test_index in kf.split(y):
    print("Kfold iteration...")
    model = MultinomialNB()
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform([' '.join(corpus[i]) for i in train_index])
    X_test = vectorizer.transform([' '.join(corpus[i]) for i in test_index])
    y_train = y[train_index]
    y_test = y[test_index]
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    model.fit(X_res, y_res)
    train_accuracy.append(model.score(X_res, y_res))
    test_accuracy.append(model.score(X_test, y_test))
    #print("Training accuracy : {}".format(model.score(X_train, y_train)))
    #print("Test accuracy : {}".format(model.score(X_test, y_test)))
    #print("Classification report for test set")
    #print(classification_report(y_test, model.predict(X_test)))

with open('SMOTE_nb_results.txt', 'w') as f:
    f.write("Train accuracy : {} \n".format(np.mean(train_accuracy)))
    f.write("Test accuracy : {} \n".format(np.mean(test_accuracy)))
    f.write(str(test_accuracy))

print("Train accuracy : {}".format(np.mean(train_accuracy)))
print("Test accuracy : {}".format(np.mean(test_accuracy)))
print(test_accuracy)

model = MultinomialNB()
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform([' '.join(i) for i in corpus])
y_train = np.array([tag for tag in corpus.iterTags()])
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
model.fit(X_res, y_res)

X, y = utils.dbUtils.getFakeNewsNet()
X_test = vectorizer.transform(X)
with open('SMOTE_nb_results.txt', 'a') as f:
    f.write("Test accuracy : {} \n".format(model.score(X_test, y)))
    f.write(str(classification_report(y, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)))