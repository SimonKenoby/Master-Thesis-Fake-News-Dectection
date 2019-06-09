#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:47:56 2018

@author: simon
"""

import numpy as np
import os
import json
import pandas as pd
import gc

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer


from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.TFE
collection = db.news

nNews = collection.count({'$or' : [{'type' : 'reliable'}, {'type' : 'fake'}]})

news = []
tags = []
print("Loading fake and reliable news")
for new in collection.find({'type' : 'fake'}).limit(200):
    news.append(new['content'])
    tags.append(new['type'])

for new in collection.find({'type' : 'reliable'}).limit(200):
    news.append(new['content'])
    tags.append(new['type'])


print("Computing tf-idf matrix")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news)
y = np.array(tags)

news = None
gc.collect()


print("10 folds corss validation on model")
model = MultinomialNB()

train_accuracy = []
test_accuracy = []
kf = KFold(n_splits=10, shuffle = True)
for train_index, test_index in kf.split(X):    
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    model.fit(X_train, y_train)
    train_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))
    #print("Training accuracy : {}".format(model.score(X_train, y_train)))
    #print("Test accuracy : {}".format(model.score(X_test, y_test)))
    #print("Classification report for test set")
    #print(classification_report(y_test, model.predict(X_test)))
print("Train accuracy : {}".format(np.mean(train_accuracy)))
print("Test accuracy : {}".format(np.mean(test_accuracy)))
print(test_accuracy)

print("Testing on different dataset")

fake_directories = ['../Data/FakeNewsNet-master/Data/BuzzFeed/FakeNewsContent', '../Data/FakeNewsNet-master/Data/PolitiFact/FakeNewsContent']
real_directories = ['../Data/FakeNewsNet-master/Data/BuzzFeed/RealNewsContent', '../Data/FakeNewsNet-master/Data/PolitiFact/RealNewsContent']

fake_files_list = []
for fake_dir in fake_directories:
    for root, directory, files in os.walk(fake_dir):
        for name in files:
            fake_files_list.append(os.path.join(root, name))
real_files_list = []
for real_dir in real_directories:
    for root, directory, files in os.walk(real_dir):
        for name in files:
            real_files_list.append(os.path.join(root, name))
            
# Open the first file in order to retreive dictionary keys
with open(fake_files_list[0]) as f:
    j = json.loads(f.read())
keys = j.keys()
data = pd.DataFrame(columns=keys)
for file_name in fake_files_list:
    with open(file_name) as f:
        j = json.loads(f.read())
        j['type'] = 'fake'
        data = data.append(j, ignore_index=True)
for file_name in real_files_list:
    with open(file_name) as f:
        j = json.loads(f.read())
        j['type'] = 'realiable'
        data = data.append(j, ignore_index=True)   
        
new_text = data['text'].values
newy = data['type'].values

X_test = vectorizer.transform(new_text)
print("Test accuracy : {}".format(model.score(X_test, newy)))   

fake = []
for new in collection.find({'type' : 'fake'}).limit(200):
    words_list = {}
    for word in vectorizer.get_feature_names():
        if word in new['content']:
            words_list[word] = True
        else:
            words_list[word] = False
    fake.append(words_list)
    
reliable = []
for new in collection.find({'type' : 'reliable'}).limit(200):
    words_list = {}
    for word in vectorizer.get_feature_names():
        if word in new['content']:
            words_list[word] = True
        else:
            words_list[word] = False
    reliable.append(words_list)
