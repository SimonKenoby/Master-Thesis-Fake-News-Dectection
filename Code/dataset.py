#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:19:30 2018

@author: simon
"""

from __future__ import print_function, division
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pymongo import MongoClient

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class NewsDataSet(Dataset):
    def __init__(self, what):
        self.what = what
        client = MongoClient('localhost', 27017)
        self.db = client.TFE
        self.ids = []
        for obj in self.db.news.find({'$or' : [{'type' : 'reliable'}, {'type' : 'fake'}]}):
            self.ids.append(obj['_id'])
            
    def __len__(self):
            return len(self.ids)
        
    def __getitem__(self, idx):
            if self.what == 'tokenized_text':
                if self.isFake(idx):
                    return self.db.fake.find_one({'_id' : self.ids[idx]})['tokenized_text']
                else:
                    return self.db.reliable.find_one({'_id' : self.ids[idx]})['tokenized_text']
            
    def isFake(self, idx):
            return self.db.news.find_one({'_id' : self.ids[idx]})['type'] == 'fake'
        
                
        