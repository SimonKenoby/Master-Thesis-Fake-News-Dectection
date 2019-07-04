from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import TimeDistributed
from keras.utils import Sequence
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
import numpy as np
import argparse


from w2vVectorizer import text2vec

def makeModel():
	model = Sequential()
	model.add(LSTM(300, return_sequences=False, input_shape=(None, 300)))
	#model.add(Activation('softmax'))
	#model.add(TimeDistributed(Dense(1)))
	#model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	return model

class generator(Sequence):
	def __init__(self, X, y, path):
		self.X = X
		self.y = y
		self.size = len(y)
		self.vectorizer = text2vec(path = path)
		
	def __len__(self):
		return self.size
	
	def __getitem__(self, idx):
		return self.vectorizer.transform(self.X[idx]).reshape(1, -1, 300), self.y[idx]


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Training Lasso model')
	parser.add_argument('train', type=str, help = "Train set dir")
	parser.add_argument('outmodel', type=str, help = "Output file for model")
	parser.add_argument('word2vec', type=str, help = "Path to word2vec gz file")

	args = parser.parse_args()
	print(args.word2vec)


	print("Loading data")
	data = pd.read_json(args.train, lines=True)
	train = data['tokenized_text'][0:64]
	y_train = np.array(data['type'][0:64], dtype=str)

	lb = LabelBinarizer()
	lb.fit(y_train)
	y_train = lb.transform(y_train)

	print("Loading word2vec")
	gen = generator(train, y_train, args.word2vec)

	model = makeModel()
	print("Training model")
	model.fit_generator(gen, steps_per_epoch=len(y_train), epochs=10, verbose=1, max_queue_size=1, workers=1, use_multiprocessing=True)
	model.save(args.outmodel)