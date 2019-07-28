import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import gluonnlp
from mxnet.io import NDArrayIter
from tqdm import tqdm

import json
import argparse
import pandas as pd
import os
import sys
import numpy as np
from gensim.corpora import Dictionary
from sklearn.metrics import recall_score, confusion_matrix

def load_data(trainFile, dct, ctx = mx.cpu(0)):
	labels = []
	num_lines = sum(1 for line in open(trainFile))
	array = nd.ones((num_lines, SEQ_LENGTH), dtype='float32', ctx = ctx)
	print("Loading data: ")
	pbar = tqdm(total = num_lines)
	with open(trainFile) as f:
		for i, line in enumerate(f):
			l = json.loads(line)
			text = l['tokenized_text']
			label = l['type']
			labels.append(label)
			array[i] = tokens_to_idx(text, dct)
			pbar.update(1)
	pbar.close()
	return array, label_binarize(labels, ctx)

def tokens_to_idx(tokens, dct, ctx = mx.cpu(0)):
    array = [dct.token2id[token] if token in dct.token2id else -1 for token in tokens]
    if len(array) > SEQ_LENGTH:
        array = array[0:SEQ_LENGTH]
    else:
        array.extend([-1 for i in range(0, SEQ_LENGTH - len(array))])
    return nd.array(array, ctx = ctx)

def label_binarize(labels, ctx = mx.cpu(0)):
	lab = nd.zeros(len(labels), ctx = ctx)
	for i, label in enumerate(labels):
		if label == 'fake':
			lab[i] = 1
	return lab

def recall(y, y_hat):
	y = y.asnumpy()
	y_hat = y_hat.asnumpy()
	return recall_score(y, y_hat), confusion_matrix(y, y_hat).ravel()


class LSTM(gluon.Block):
	def __init__(self, vocab_size, num_embed, num_hidden, num_layers, dropout, **kwargs):
		super(LSTM, self).__init__(**kwargs)
		with self.name_scope():
			self.encoder = gluon.nn.Embedding(vocab_size, num_embed)
			self.LSTM1 = gluon.rnn.LSTM(num_embed, num_layers, layout = 'NTC', bidirectional = True)
			self.dropout = gluon.nn.Dropout(dropout)
			self.fc1 = gluon.nn.Dense(1, activation='sigmoid')
			
	def forward(self, inputs, hidden):
		emb = self.encoder(inputs)
		output, hidden = self.LSTM1(emb, hidden)
		output = self.dropout(output)
		output = self.fc1(output)
		return output, hidden
	
	def begin_state(self, *args, **kwargs):
		return self.LSTM1.begin_state(*args, **kwargs)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments for LSTM model')
	parser.add_argument('--test', nargs='+', type=str, help = "Validation set file", required=True)
	parser.add_argument('--input', type=str, help = "Input directory for the model files")
	parser.add_argument('--dictFile', type=str, help = "Path to the dictionary file")
	parser.add_argument('--SEQ_LENGTH', type = int, help = "Fixed size length to expand or srink text")
	parser.add_argument('--EMBEDDING_DIM', type = int, help = "Size of the embedding dimention")
	parser.add_argument('--HIDDEN', type = int, help = "Size of the hidden layer")
	parser.add_argument('--LAYERS', type = int, help = "Number of hidden layers")
	parser.add_argument('--DROPOUT', type = float, help = "Number of hidden layers")
	parser.add_argument('--BATCH_SIZE', type = int, help = "Batch size")
	parser.add_argument('--utils', type=str, help = "Helper directory")
	parser.add_argument('--db', type=str, help = "DB name", required=True)
	parser.add_argument('--collection', type=str, help = "DB collection")
	parser.add_argument('--host', type=str, help = "DB host")
	parser.add_argument('--port', type=int, help = "Port number of db")

	args = parser.parse_args()

	sys.path.append(args.utils)

	from register_experiment import Register

	testFiles = args.test

	SEQ_LENGTH = args.SEQ_LENGTH
	EMBEDDING_DIM = args.EMBEDDING_DIM
	HIDDEN = args.HIDDEN
	LAYERS = args.LAYERS
	DROPOUT = args.DROPOUT
	BATCH_SIZE = args.BATCH_SIZE
	ctx = mx.gpu(1)

	files = []
	# r=root, d=directories, f = files
	for r, d, f in os.walk(args.input):
		for file in f:
			files.append(os.path.join(r, file))
	files.sort()

	r = Register(args.host, args.port, args.db, args.collection)
	print(r.getLastExperiment())

	pbar = tqdm(len(testFiles))
	for i, test_file in enumerate(testFiles):
		dct = Dictionary.load(args.dictFile)
		array, labels = load_data(test_file, dct)
		acc = mx.metric.Accuracy()
		accuracy = []
		for j, model in enumerate(files):
			recall_list = []
			cfMatrix = []
			net = LSTM(len(dct), EMBEDDING_DIM, HIDDEN, LAYERS, DROPOUT)
			net.load_parameters(model, ctx=ctx)
			hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx = ctx)
			nd_iter = NDArrayIter(data={'data':array},
							  label={'softmax_label':labels},
							  batch_size=BATCH_SIZE)
			for batch in nd_iter:
				output, _ = net(batch.data[0].copyto(ctx), hidden)
				pred = output > 0.5
				y = batch.label[0]
				acc.update(y, pred)
				rec, mat = recall(y, pred)
				recall_list.append(rec)
				cfMatrix.append(mat)
			accuracy.append(acc.get()[1])
			r.addEpochs(j, {'accuracy' : acc.get()[1], 'recall' : np.mean(recall_list), 'Confusion Matrix' : list(map(int, sum(cfMatrix)))}, r.getLastExperiment() + 1, 'valid')
		pbar.update(1)
	pbar.close()
	r.closeExperiment(r.getLastExperiment() + 1)
		