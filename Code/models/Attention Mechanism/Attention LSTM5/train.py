import mxnet.gluon as gluon
import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon.data import DataLoader
import gluonnlp
import argparse
import numpy as np
import json
import sys


class train(gluon.data.Dataset):
    def __init__(self, split, SEQ_LENGTH, EMBEDDING_DIM, utils, ctx = mx.cpu(0), **kwargs):
        super(train, self).__init__(**kwargs)
        sys.path.append(utils)
        import dbUtils
        self.utils_path = utils
        self.SEQ_LENGTH = SEQ_LENGTH
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.ctx = ctx
        self.embed = gluonnlp.embedding.create(embedding_name='word2vec', source="GoogleNews-vectors-negative300")
        self.fake = dbUtils.TokenizedIterator('news_cleaned', filters = {'split' : 'train', 'type' : 'fake'})
        self.reliable = dbUtils.TokenizedIterator('news_cleaned', filters = {'split' : 'train', 'type' : 'reliable'})
        self.switch = True
        
    def __getitem__(self, index):

        return self.load_data(index)

    def __len__(self):
        return 2*len(self.reliable)

    def load_data(self, index):
        array = nd.zeros(shape=(1, self.SEQ_LENGTH, self.EMBEDDING_DIM), dtype='float32', ctx = self.ctx)
        if self.switch:
            text = self.fake[(index // 2) % len(self.fake)]
            self.switch = False
            label = 'fake'
        else:
            text = self.reliable[index // 2]
            self.switch = True
            label = 'reliable'
        if len(text) > self.SEQ_LENGTH:
            text = text[0:self.SEQ_LENGTH]
        else:
            text.extend(['<PAD>' for i in range(0, self.SEQ_LENGTH - len(text))])
        array = self.embed[text]
        label = self.label_binarize(label)
        return array, label

    def label_binarize(self, labels):
        if labels == 'fake':
            return nd.ones(shape = 1, dtype='float32')
        return nd.zeros(shape = 1, dtype='float32')
            

    
if __name__ == "__main__":

    ds = train('train', 20, 300, '/home/simon/Documents/TFE/Code/utils')
    print(len(ds))

    dl = DataLoader(ds, batch_size = 32)

    for batch, label in dl:
        print(batch.shape)
        print(label)
        break