#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:41:39 2019

@author: simon
"""

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.TFE
collection = db.news_copy

idx = 0

for obj in collection.find({}, {'_id' : True}):
    collection.update_one({'_id' : obj['_id']}, {'$set' : {'idx' : idx}})
    idx += 1