import sys

import mxnet as mx
import mxnet.gluon as gluon
import mxnet.ndarray as nd
from mxnet.gluon.data import DataLoader
from utils import dbUtils
from gensim.corpora import Dictionary

class dataset1(gluon.data.Dataset):
    def __init__(self, dataset, filters, SEQ_LENGTH, dct, ctx=mx.cpu(0), **kwargs):
        super(dataset1, self).__init__(**kwargs)
        self.SEQ_LENGTH = SEQ_LENGTH
        self.ctx = ctx
        self.fake = dbUtils.TokenizedIterator(dataset, filters={**filters, **{'type': 'fake'}})
        self.reliable = dbUtils.TokenizedIterator(dataset, filters={**filters, **{'type': 'reliable'}})
        self.switch = True
        self.dct = Dictionary.load(dct)

    def __getitem__(self, index):
        return self.load_data(index)

    def __len__(self):
        if len(self.reliable) >= len(self.fake):
            return 2 * len(self.reliable)
        else:
            return 2 * len(self.fake)

    def load_data(self, index):
        if self.switch:
            text = self.fake[(index // 2) % len(self.fake)]
            self.switch = False
            label = 'fake'
        else:
            text = self.reliable[index // 2 % len(self.reliable)]
            self.switch = True
            label = 'reliable'
        array = self.tokens_to_idx(text)
        label = self.label_binarize(label)
        return array, label

    def tokens_to_idx(self, tokens, ctx=mx.cpu(0)):
        array = [self.dct.token2id[token] if token in self.dct.token2id else -1 for token in tokens]
        if len(array) > self.SEQ_LENGTH:
            array = array[0:self.SEQ_LENGTH]
        else:
            array.extend([-1 for i in range(0, self.SEQ_LENGTH - len(array))])
        return nd.array(array, ctx=ctx, dtype='int32')

    def label_binarize(self, label, ctx=mx.cpu(0)):
        lab = nd.zeros(shape=(2), ctx=ctx)
        if label == 'fake':
            lab[1] = 1
        else:
            lab[0] = 1
        return lab

    def dctLen(self):
        return len(self.dct)


if __name__ == "__main__":

    ds = dataset1('news_cleaned', {'split' : 'valid'}, 10, 'Dictionary.dct')
    print(len(ds))

    dl = DataLoader(ds, batch_size=32)

    for batch, label in dl:
        print(batch.shape)
        print(label)
        break
