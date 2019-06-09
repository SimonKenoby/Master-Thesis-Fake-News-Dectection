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


        
client = MongoClient('localhost', 27017)
db = client.TFE
collection = db.news_cleaned


pbar = tqdm(total=collection.count_documents({}))

table = str.maketrans('', '', string.punctuation)
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_numeric, 
                    remove_stopwords, strip_multiple_whitespaces, strip_non_alphanum, strip_punctuation2,
                    lambda x: re.sub('\s+', ' ', x), lambda x: re.sub("\'", "", x), lambda x: x.translate(table), strip_short]

for news in collection.find():
    text = news['content']
    sentLength, num_sentences = averageSentence(text)
    text = preprocess_string(text, CUSTOM_FILTERS)
    length = len(text)
    collection.update_one({'_id' : news['_id']}, {'$set' : {'tokenized_text' : text, 'word_count' : length, 'avgSentenceLength' : sentLength, 'numSentences' : num_sentences}})
    pbar.update(1)


pbar.close()
    
