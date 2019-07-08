import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import gluonnlp
from mxnet.io import NDArrayIter
from mxnet import autograd
from mxboard import *

from tqdm import tqdm

import json
import argparse
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt

def load_data(trainFile, ctx = mx.cpu(0)):
	embed = gluonnlp.embedding.create(embedding_name='word2vec', source="GoogleNews-vectors-negative300")
	labels = []
	num_lines = sum(1 for line in open(trainFile))
	array = nd.ones((num_lines, SEQ_LENGTH, EMBEDDING_DIM), dtype='float32', ctx = ctx)
	print("Loading data: ")
	pbar = tqdm(total = num_lines)
	with open(trainFile) as f:
		for i, line in enumerate(f):
			l = json.loads(line)
			text = l['tokenized_text']
			label = l['type']
			labels.append(label)
			if len(text) > SEQ_LENGTH:
				text = text[0:SEQ_LENGTH]
			else:
				text.extend(['<PAD>' for i in range(0, SEQ_LENGTH - len(text))])
			array[i] = embed[text]
			pbar.update(1)
	pbar.close()
	return array, label_binarize(labels, ctx)

def label_binarize(labels, ctx = mx.cpu(0)):
	lab = nd.zeros(len(labels), ctx = ctx)
	for i, label in enumerate(labels):
		if label == 'fake':
			lab[i] = 1
	return lab

def Evaluate(net, X, y, ctx = mx.cpu(0)):
	hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx = mx.gpu(0))
	output, hidden = net(X.copyto(mx.gpu(0)), hidden)
	acc = mx.metric.Accuracy()
	acc.update(output.copyto(mx.cpu(0)), y.flatten())
	return acc.get()



class LSTM(gluon.Block):
	def __init__(self, vocab_size, num_embed, num_hidden, num_layers, dropout, **kwargs):
		super(LSTM, self).__init__(**kwargs)
		with self.name_scope():
			self.LSTM1 = gluon.rnn.LSTM(num_embed, num_layers, layout = 'NTC', bidirectional = True)
			self.drop1 = gluon.nn.Dropout(dropout)
			self.fc1 = gluon.nn.Dense(1, activation='sigmoid')
			
	def forward(self, inputs, hidden):
		output, hidden = self.LSTM1(inputs, hidden)
		output = self.fc1(output)
		return output, hidden
	
	def begin_state(self, *args, **kwargs):
		return self.LSTM1.begin_state(*args, **kwargs)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments for LSTM model')
	parser.add_argument('train', type=str, help = "Train set file")
	parser.add_argument('test', type=str, help = "Validation set file")
	parser.add_argument('outmodel', type=str, help = "Output file for model")
	parser.add_argument('word2vec', type=str, help = "Path to word2vec gz file")
	parser.add_argument('dictFile', type=str, help = "Path to the dictionary file")
	parser.add_argument('SEQ_LENGTH', type = int, help = "Fixed size length to expand or srink text")
	parser.add_argument('EMBEDDING_DIM', type = int, help = "Size of the embedding dimention")
	parser.add_argument('HIDDEN', type = int, help = "Size of the hidden layer")
	parser.add_argument('LAYERS', type = int, help = "Number of hidden layers")
	parser.add_argument('BATCH_SIZE', type = int, help = "Batch size")
	parser.add_argument('EPOCHS', type = int, help = "Number of epochs for training")
	parser.add_argument('log', type=str, help = "Log Directory")

	args = parser.parse_args()

	dictFile = args.dictFile
	trainFile = args.train
	testFile = args.test

	SEQ_LENGTH = args.SEQ_LENGTH
	EMBEDDING_DIM = args.EMBEDDING_DIM
	HIDDEN = args.HIDDEN
	LAYERS = args.LAYERS
	BATCH_SIZE = args.BATCH_SIZE
	EPOCHS = args.EPOCHS
	LOG = args.log
	ctx = mx.gpu(0)

	sw = SummaryWriter(logdir=LOG, flush_secs=5)

	mx.random.seed(42, ctx = mx.cpu(0))
	mx.random.seed(42, ctx = mx.gpu(0))

	array, labels = load_data(trainFile)
	#X_test, y_test = load_data(testFile)

	net = LSTM(len(array), EMBEDDING_DIM, HIDDEN, LAYERS, 0.2)
	net.initialize(mx.init.Normal(sigma=1), ctx = ctx)
	trainer = gluon.Trainer(net.collect_params(), 'rmsprop', {'learning_rate': 0.01})
	loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
	hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx = mx.cpu(0))

	for epochs in range(0, EPOCHS):
		pbar = tqdm(total = len(array) // BATCH_SIZE)
		total_L = 0.0
		accuracy = []
		hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx = ctx)
		nd_iter = NDArrayIter(data={'data':array},
							  label={'softmax_label':labels},
							  batch_size=BATCH_SIZE)
		acc = mx.metric.Accuracy()
		for batch in nd_iter:
			pbar.update(1)
			with autograd.record():
				output, hidden = net(batch.data[0].copyto(ctx), hidden)
				L = loss(output, batch.label[0].copyto(ctx))
				L.backward()
				acc.update(output, batch.label[0].flatten())
			trainer.step(BATCH_SIZE)
			total_L += mx.nd.sum(L).asscalar()
		#test.append(Evaluate(net, X_test, y_test))
		pbar.close() 

		# TODO: Print epoch number
		# TODO: Evalute on validation set at the same time
		# TODO: Make plots of training 
		print("Loss : {}, Accuracy : {}".format(total_L, acc.get()))
		sw.add_scalar(tag = 'accuracy', value = acc.get()[1], global_step = epochs)
		sw.add_scalar(tag = 'loss', value = total_L, global_step = epochs)
	sw.close()

	net.save_parameters(args.outmodel)