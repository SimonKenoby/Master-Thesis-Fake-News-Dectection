#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:57:40 2018

@author: simon
"""

import re
import string
import numpy as np


from pymongo import MongoClient
from queue import Queue
from threading import Thread
from tqdm import tqdm

from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, remove_stopwords, strip_numeric, strip_short, strip_multiple_whitespaces, strip_non_alphanum, strip_punctuation2
from gensim.corpora.dictionary import Dictionary

from nltk import tokenize




def averageSentence(text):
    sentences = tokenize.sent_tokenize(text)
    length = []
    for sentence in sentences:
        words = tokenize.word_tokenize(sentence.translate(str.maketrans('','',string.punctuation)))
        length.append(len(words))
    return np.mean(length), len(sentences)


class worker(Thread):
    def __init__(self, number, q, pbar):
        Thread.__init__(self)
        self.number = number
        self.q = q
        self.pbar = pbar
        
    def run(self):
        print("Thread {} started".format(self.number))
        client = MongoClient('localhost', 27017)
        db = client.TFE
        collection = db.news_cleaned
        i = 0
        table = str.maketrans('', '', string.punctuation)
        CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_numeric, 
                          remove_stopwords, strip_multiple_whitespaces, strip_non_alphanum, strip_punctuation2,
                          lambda x: re.sub('\s+', ' ', x), lambda x: re.sub("\'", "", x), lambda x: x.translate(table), strip_short]
        while not self.q.empty():
            news_id = self.q.get()
            news = collection.find_one({'_id' : news_id})
            text = news['content']
            sentLength, num_sentences = averageSentence(text)
            text = preprocess_string(text, CUSTOM_FILTERS)
            length = len(text)
            collection.update_one({'_id' : news_id}, {'$set' : {'tokenized_text' : text, 'word_count' : length, 'avgSentenceLength' : sentLength, 'numSentences' : num_sentences}})
            i += 1
            self.pbar.update(1)
       
        
client = MongoClient('localhost', 27017)
db = client.TFE
collection = db.news_cleaned

ids = Queue()

for obj in collection.find():
    ids.put(obj['_id'])
         
threads = []
pbar = tqdm(total=ids.qsize())

for i in range(0, 10):
    threads.append(worker(i, ids, pbar))
    
for t in threads:
    t.start()
    
for t in threads:
    t.join()

pbar.close()
    

'''
word_count = []
for obj in db.fake.find():
    word_count.append(len(obj['tokenized_text']))

for obj in db.reliable.find():
    word_count.append(len(obj['tokenized_text']))
    
def build_dictionary():
    client = MongoClient('localhost', 27017)
    db = client.TFE
    dic = Dictionary()
    documents = []
    for document in db.fake.find().limit(200):
        documents.append(document['tokenized_text'])
    dic.add_documents(documents)
    documents = []
    for document in db.reliable.find().limit(200):
        documents.append(document['tokenized_text'])
    dic.add_documents(documents)
    return dic
'''