#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:42:49 2018

@author: simon
"""

import numpy as np
import string
from nltk import tokenize
from pymongo import MongoClient
from queue import Queue
from threading import Thread


def averageSentence(text):
    sentences = tokenize.sent_tokenize(text)
    length = []
    for sentence in sentences:
        words = tokenize.word_tokenize(sentence.translate(str.maketrans('','',string.punctuation)))
        length.append(len(words))
    return np.mean(length), len(sentences)


client = MongoClient('localhost', 27017)
db = client.TFE
collection = db.news

ids = Queue()

for obj in collection.find({ '$or' : [{'avgSentenceLength' : {'$exists' : False}}, {'numSentences' : {'$exists' : False}}]}):
    ids.put(obj['_id'])
    


class worker(Thread):
    def __init__(self, number, q):
        Thread.__init__(self)
        self.number = number
        self.q = q
        
    def run(self):
        print("Thread {} started".format(self.number))
        client = MongoClient('localhost', 27017)
        db = client.TFE
        collection = db.news_copy
        i = 0
        while not self.q.empty():
            news_id = self.q.get()
            news = collection.find_one({'_id' : news_id})
            length, num_sentences = averageSentence(news['content'])
            collection.update_one({'_id' : news_id}, {'$set' : {'avgSentenceLength' : length, 'numSentences' : num_sentences}})
            i += 1
            if i % 1000 == 0:
                print("Thread {} did {} jobs".format(self.number, i))
                
threads = []

for i in range(0, 10):
    threads.append(worker(i, ids))
    
for t in threads:
    t.start()
    
for t in threads:
    t.join()


