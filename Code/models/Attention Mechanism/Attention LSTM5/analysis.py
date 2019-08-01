#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 08:50:15 2019

@author: simon
"""

from pymongo import MongoClient
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


client = MongoClient('localhost', 27017)
db = client.TFE
collection = db.results3

res = collection.find_one({'experiment_id' : 1})

def generateFromCM(cm):
    tn, fp, fn, tp = cm
    y_true = ['reliable' for i in range(0, tn + fp)]
    y_pred = ['reliable' for i in range(0, tn)]+['fake' for i in range(0, fp)]
    y_true = y_true + ['fake' for i in range(0, fn + tp)]
    y_pred = y_pred + ['reliable' for i in range(0, fn)] + ['fake' for i in range(0, tp)]
    return y_true, y_pred

def aggregateTest(epoch):
    cm = np.array([0, 0, 0, 0])
    for res in epoch['valid']:
        cm += np.array(res['Confusion Matrix'])
    return cm

cm = aggregateTest(res['result'][20])

y_true, y_pred = generateFromCM(cm)
clr = classification_report(y_true, y_pred, output_dict = True)

clr_list = []
for result in res['result']:
    cm = aggregateTest(result)
    y_true, y_pred = generateFromCM(cm)
    clr_list.append(classification_report(y_true, y_pred, output_dict = True))
    
sorted(clr_list, key = lambda x: x['weighted avg']['f1-score'])