#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 08:33:36 2019

@author: simon
"""

from pymongo import MongoClient
import numpy as np
from tqdm import tqdm

client = MongoClient('localhost', 27017)
db = client.TFE
collection = db.news_cleaned


fake = collection.distinct('domain', {'type' : 'fake', 'domain' : {'$nin' : ['beforeitsnews.com']}})

pbar = tqdm(total = len(fake))
for domain in fake:
    res = [res['_id'] for res in collection.find({'domain' : domain})]
    np.random.shuffle(res)
    a, b, c = np.split(res, [int(0.6*len(res)), int(0.6*len(res)) + int(0.2 * len(res))])
    for idx in a:
        collection.update_one({'_id' : int(float(idx))}, {'$set' : {'split' : 'train'}})
    for idx in b:
        collection.update_one({'_id' : int(float(idx))}, {'$set' : {'split' : 'valid'}})
    for idx in c:
        collection.update_one({'_id' : int(float(idx))}, {'$set' : {'split' : 'test'}})
    pbar.update(1)
pbar.close()
        
reliable = collection.distinct('domain', {'type' : 'reliable', 'domain' : {'$nin' : ['nytimes.com']}})

pbar = tqdm(total = len(reliable))

for domain in reliable:
    res = [res['_id'] for res in collection.find({'domain' : domain})]
    np.random.shuffle(res)
    a, b, c = np.split(res, [int(0.6*len(res)), int(0.6*len(res)) + int(0.2 * len(res))])
    for idx in a:
        collection.update_one({'_id' : int(float(idx))}, {'$set' : {'split' : 'train'}})
    for idx in b:
        collection.update_one({'_id' : int(float(idx))}, {'$set' : {'split' : 'valid'}})
    for idx in c:
        collection.update_one({'_id' : int(float(idx))}, {'$set' : {'split' : 'test'}})
    pbar.update(1)
pbar.close()