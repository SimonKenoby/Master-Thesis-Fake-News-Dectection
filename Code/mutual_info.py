#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:11:14 2018

@author: simon
"""

from gensim.corpora.dictionary import Dictionary
from pymongo import MongoClient
import math

client = MongoClient('localhost', 27017)
db = client.TFE

infos = {'fake_count' : db.fake.find().count(), 'reliable_count' : db.reliable.find().count()}

fake_dic = Dictionary()
full_dic = Dictionary()
reliable_dic = Dictionary()
fake_texts = []
for doc in db.fake.find():
    fake_texts.append(doc['tokenized_text'])
    
fake_dic.add_documents(fake_texts)
full_dic.add_documents(fake_texts)
fake_texts = []

reliable_texts = []
for doc in db.reliable.find():
    reliable_texts.append(doc['tokenized_text'])
    
reliable_dic.add_documents(reliable_texts)
full_dic.add_documents(reliable_texts)
reliable_texts = []

infos['full_dic'] = full_dic
infos['reliable_dic'] = reliable_dic
infos['fake_dic'] = fake_dic

def MU(word, infos):
    total = infos['fake_count'] + infos['reliable_count']
    proba = []
    if word in infos['fake_dic'].values():
        word_id = infos['fake_dic'].token2id[word]
        proba.append(infos['fake_dic'].dfs[word_id] / infos['fake_count'])
        proba.append(1 - proba[0])
    else:
        proba.append(0)
        proba.append(1)
    if word in infos['reliable_dic'].values():
        word_id = infos['reliable_dic'].token2id[word]
        proba.append(infos['reliable_dic'].dfs[word_id] / infos['reliable_count'])
        proba.append(1 - proba[2])
    else:
        proba.append(0)
        proba.append(1)
    pfake = infos['fake_count'] / total
    preliable = infos['reliable_count'] / total
    word_id = infos['full_dic'].token2id[word]
    pPresent = infos['full_dic'].dfs[word_id] / total
    pAbs = 1 - pPresent
    MU = []
    if proba[0] != 0:
        MU.append(proba[0] * math.log2(proba[0] / (pfake * pPresent)))
    if proba[1] != 0:
        MU.append(proba[1] * math.log2(proba[1] / (pfake * pAbs)))
    if proba[2] != 0:
        MU.append(proba[2] *  math.log2(proba[2] / (preliable * pPresent)))
    if proba[3] != 0:
        MU.append(proba[3] *  math.log2(proba[3] / (preliable * pAbs)))
    x = 0
    for p in MU:
        x += p
    return x

Mutual_info = {}
for word in full_dic.values():
    Mutual_info[word] = MU(word, infos)
    