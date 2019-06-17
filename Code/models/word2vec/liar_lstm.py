#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:38:40 2019

@author: simon
"""

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
from sklearn.metrics import classification_report
import numpy as np

from w2vVectorizer import text2vec

from pymongo import MongoClient
import datetime

import sys
sys.path.append('../..')

import utils.dbUtils
import utils.gensimUtils

data = utils.dbUtils.TokenizedIterator('liar_liar', filters = {'split' : 'train'})
train = [value for value in data]
y_train = np.array([x for x in data.iterTags()])

lb = LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)



class generator(Sequence):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.size = len(y)
        self.vectorizer = text2vec()
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.vectorizer.transform(self.X[idx]).reshape(1, -1, 300), self.y[idx]
    
gen = generator(train, y_train)

model = Sequential()
model.add(LSTM(300, return_sequences=False, input_shape=(None, 300)))
#model.add(Activation('softmax'))
#model.add(TimeDistributed(Dense(1)))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(gen, steps_per_epoch=len(y_train), epochs=10, verbose=1, max_queue_size=16, workers=8, use_multiprocessing=True)

model.save("lstm.h5")


data = utils.dbUtils.TokenizedIterator('liar_liar', filters = {'split' : 'test'})
valid = [value for value in data]
y_valid = np.array([x for x in data.iterTags()])

y_valid = lb.transform(y_valid)
gen = generator(valid, y_valid)

y_hat = []
for i in range(0, len(y_valid)):
    y_hat.append(model.predict_classes(gen[i][0])[0][0])
    
clr = classification_report(y_valid.flatten(), y_hat, output_dict = True)

client = MongoClient('192.168.178.25', 27017)
db = client.TFE
collection = db.results

idx = collection.insert_one({'date' : datetime.datetime.now(), 'corpus' : 'liar_liar', 'experiment_id' : 1})
collection.update_one({'_id' : idx.inserted_id},{'$set' : {'model' : 'LSTM', 'classification_report' : clr}})
collection.update_one({'_id' : idx.inserted_id},{'$set' : {'model_json' : model.to_json()}})

