#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:01:39 2019

@author: simon
"""

from DataLoader import DataLoader
import mxnet.ndarray as nd

TRAIN = "/home/simon/Documents/TFE/Data/train.json"
word2vec = "/home/simon/Documents/TFE/Data/word2vec-google-news-300.gz"

dl = DataLoader(word2vec, 1, 5, 300, padding=nd.zeros(shape=300), unknown=nd.random_normal(shape=(300,)), train_file=TRAIN)

for batch in dl.generate():
    print(batch)
    break