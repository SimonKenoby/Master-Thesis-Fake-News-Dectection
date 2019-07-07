import pandas as pd
import gensim
import mxnet.ndarray as nd

class DataLoader():
	def __init__(self, word2vecFile, BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE, padding = None, unknown = None, train_file = None, test_file = None, mode = 'train'):
		self.train_file = train_file
		self.test_file = test_file
		self.word2vecFile = word2vecFile
		self.BATCH_SIZE = BATCH_SIZE
		self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
		self.EMBEDDING_SIZE = EMBEDDING_SIZE
		self.padding = padding
		self.unknown = unknown
		self.mode = mode

		if mode == 'train':
			self.data = pd.read_json(self.train_file, lines = True)
		elif mode == 'test':
			self.data = pd.read_json(self.test_file, lines = True)

		self.model = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(self.word2vecFile, binary=True)

		self.size = len(self.data)
		self.batches = self.size // self.BATCH_SIZE

	def setMode(mode):
		self.mode = mode
		if mode == 'train':
			self.data = pd.read_json(self.train_file, lines = True)
		elif mode == 'test':
			self.data = pd.read_json(self.test_file, lines = True)

		self.size = len(self.data)

	def generate(self):
		for i in range(0, self.batches):
			batch = nd.ones((self.BATCH_SIZE, self.SEQUENCE_LENGTH, self.EMBEDDING_SIZE))
			pint(batch.shape)
			for j in range(0, self.BATCH_SIZE):
				k = i * self.BATCH_SIZE + j
				batch[j] = self.transform(k)
			print(batch)
			yield batch

	def transform(self, index):
		text = self.data.loc[index]['tokenized_text']
		array = nd.array((self.SEQUENCE_LENGTH, self.EMBEDDING_SIZE))
		text_length = len(text)
		if self.unknown != None:
			if self.padding != None:
				for i in range(0, self.SEQUENCE_LENGTH):
					if i < text_length:
						if text[i] in self.model:
							array[i] = nd.array(self.model[text[i]])
						else:
							array[i] = self.unknown
					else:
						array[i] = self.padding

