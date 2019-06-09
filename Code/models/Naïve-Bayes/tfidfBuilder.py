import sys
sys.path.append('../..')

import utils.gensimUtils

from gensim.models import TfidfModel

class tfidfBuilder(object):

    def __init__(self, collection, limit = 0):
        self.corpus = utils.gensimUtils.corpusBuilder(collection, limit)
        self.model = TfidfModel(self.corpus)

    def __len__(self):
        return len(self.corpus)

    def __iter__(self):
        for text in self.corpus:
            yield self.model[text]
    
    def __getitem__(self, idx):
        return self.model[self.corpus[idx]]
