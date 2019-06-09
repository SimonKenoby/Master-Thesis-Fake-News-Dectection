import numpy as np
import os
import json
import pandas as pd
import logging


import sys
sys.path.append('../..')

import utils.dbUtils
import utils.gensimUtils

import gensim.downloader as API

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class w2v_transform(object):

    def __init__(self, n_dim):
        self.model = API.load("word2vec-google-news-300")
        self.corpus = utils.dbUtils.TokenizedIterator('news_copy', filters = {'type' : {'$in' : ['fake', 'reliable']}}, limit = 200)
        self.n_dim = n_dim

    def __getitem__(self, key):
        vector = np.random.uniform(-0.25, 0.25, (self.n_dim, 300))
        for i in range(0, self.n_dim):
            try:
                vector[i, :] = self.model[self.corpus[key][i]]
            except KeyError:
                print("Word '{}' cannot be found".format(self.corpus[key][i]))
            except IndexError:
                pass
        return vector
