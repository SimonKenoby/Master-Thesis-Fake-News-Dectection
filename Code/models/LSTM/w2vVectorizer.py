import numpy as np
import gensim

class text2vec:
    def __init__(self, padding = None, unknown = 'random', path = ''):
        self.padding = None
        self.unknown = unknown
        if unknown == 'random':
            self.UNK = np.random.normal(0, 1, 300)
        print("Loading word2vec model...")
        self.model = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(path, binary=True)
        print("Model Loaded")

    def transform(self, text):
        vector = np.zeros((len(text), 300))
        for i, word in enumerate(text):
            if word in self.model:
                vector[i] = self.model[word]
            else:
                vector[i] = self.UNK
        return vector