from gensim.test.utils import datapath
from gensim.models import FastText
import numpy as np

class Vectorizer:
    def __init__(self):
        print("Loading FastText model...")
        cap_path = datapath("/home/simon/Documents/TFE/Code/models/LSTM/crawl-300d-2M-subword.bin")
        self.model = FastText.load_fasttext_format(cap_path, encoding='ISO-8859-1')

    def transform(self, text):
        UKN = np.random.randn(300)
        vector = []
        for word in text:
            if word in self.model:
                vector.append(self.model[word])
            else:
                vector.append(UKN)
            
        if len(vector) > 0:
            vector = np.array(vector)
            return vector.mean(axis=0)

class text2vec:
    def __init__(self, padding = None, unknown = 'random'):
        cap_path = datapath("/home/simon/Documents/TFE/Code/models/LSTM/crawl-300d-2M-subword.bin")
        self.padding = None
        self.unknown = unknown
        if unknown == 'random':
            self.UNK = np.random.normal(0, 1, 300)
        print("Loading word2vec model...")
        self.model = FastText.load_fasttext_format(cap_path, encoding='ISO-8859-1')
        print("Model Loaded")

    def transform(self, text):
        vector = np.zeros((len(text), 300))
        for i, word in enumerate(text):
            if word in self.model:
                vector[i] = self.model[word]
            else:
                vector[i] = self.UNK
        return vector

'''
class batchText2vec(text2vec):
    def __init__(self, padding = None, unknown = 'random', data, BATCH = 64):
        super(batchText2vec, self).__init__()
        self.BATCH = BATCH
        self.data = data

    
    def __iter__(self):
        for i in range(0, len(data)):
    

    def __gettiem__(self, batch):
        vector = np.zeros(self.BATCH, dtype=object)
        for i, j in enumerate(range(self.BATCH * batch, self.BATCH * (batch + 1)):
            vector[i] = self.transform(self.data[j])
        return vector
        '''