#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:02:22 2019

@author: simon
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import TimeDistributed
from keras.utils import Sequence
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
import numpy as np

from w2vVectorizer import text2vec

import sys

path = sys.argv[1]

print("Loading data")

data = pd.read_json(path+"train.json", lines=True)
train = data['tokenized_text'].values[0:128]
y_train = data['type'].values[0:128]

print("Label binarization")

lb = LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)



class generator(Sequence):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.size = len(y)
        self.vectorizer = text2vec(path)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.vectorizer.transform(self.X[idx]).reshape(1, -1, 300), self.y[idx]
    
print("Making generator")
gen = generator(train, y_train)

print("Making model")
model = Sequential()
model.add(LSTM(300, return_sequences=False, input_shape=(None, 300)))
#model.add(Activation('softmax'))
#model.add(TimeDistributed(Dense(1)))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Fiting model")
model.fit_generator(gen, steps_per_epoch=len(y_train), epochs=10, verbose=1, max_queue_size=1, workers=1, use_multiprocessing=True)

model.save(path+"lstm.h5")

'''
test = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'type' : {'$in' : ['fake', 'reliable']}, 'domain' : {'$in' : ['nytimes.com', 'beforeitsnews.com']}}, limit=64)
y_test = np.array([x for x in train.iterTags()])

lb = LabelBinarizer()
lb.fit(y_test)
y_train = lb.transform(y_test)

def test_generator():
    while True:
        for i, text in enumerate(test):
            yield vectorizer.transform(text).reshape(1, -1, 300), y_train[i]

score = model.evaluate_generator(test_generator(), steps = 64)
print([model.predict(vectorizer.transform(text).reshape(1, -1, 300)) for text in test])
'''