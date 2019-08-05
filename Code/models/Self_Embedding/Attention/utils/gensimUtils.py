from .dbUtils import TokenizedIterator
from gensim.corpora import Dictionary


class corpusBuilder(object):

    def __init__(self, collection, limit = 0, filters = {}):
        self.iter = TokenizedIterator(collection, limit, filters)
        self.dct = Dictionary(self.iter)

    def __len__(self):
        return len(self.iter)

    def __iter__(self):
        for obj in self.iter:
            yield self.dct.doc2bow(obj)

    def __getitem__(self, idx):
        return self.dct.doc2bow(self.iter[idx])