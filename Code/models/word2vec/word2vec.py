#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:00:17 2018

@author: simon
"""

import re
import string
import logging
import os

import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, remove_stopwords, strip_numeric, strip_short, strip_multiple_whitespaces, strip_non_alphanum, strip_punctuation2
from gensim.corpora.dictionary import Dictionary
from gensim.summarization.textcleaner import get_sentences
import gensim.downloader as API

from pymongo import MongoClient
from queue import Queue

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self):
        client = MongoClient('localhost', 27017)
        self.db = client.TFE
        self.collection = self.db.news
 
    def __iter__(self):
        table = str.maketrans('', '', string.punctuation)
        CUSTOM_FILTERS = [strip_tags, strip_punctuation, strip_numeric, 
                          remove_stopwords, strip_multiple_whitespaces, 
                          strip_non_alphanum, strip_punctuation2,
                          lambda x: re.sub('\s+', ' ', x), lambda x: re.sub("\'", "", x), 
                          lambda x: x.translate(table), strip_short]
        for obj in self.collection.find():
            for sentence in get_sentences(obj['content']):
                yield preprocess_string(sentence, CUSTOM_FILTERS)

sentences = MySentences()
model_name = 'word2vec_300_case_kept.model'
if not os.path.isfile(model_name):
    model = Word2Vec(sentences, workers=8, size=300)
    model.save(model_name)
else:
    model = Word2Vec(model_name)

for i, sent in enumerate(sentences):
    print(model.wv[sent])
    if i >= 2:
        break
