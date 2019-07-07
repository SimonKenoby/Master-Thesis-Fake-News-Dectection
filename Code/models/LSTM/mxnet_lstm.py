import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
from mxnet.io import NDArrayIter
from mxnet import autograd

import pandas as pd
import argparse
from gensim.corpora import Dictionary

def tokens_to_idx(tokens, ctx = mx.cpu(0)):
    array = [dct.token2id[token] for token in tokens]
    if len(array) > SEQ_LENGTH:
        array = array[0:SEQ_LENGTH]
    else:
        array.extend([-1 for i in range(0, SEQ_LENGTH - len(array))])
    return nd.array(array, ctx = ctx)

def data_to_array(data, ctx = mx.cpu(0)):
    array = nd.zeros((len(data), SEQ_LENGTH), ctx = ctx)
    for i, text in enumerate(data['tokenized_text']):
        array[i] = tokens_to_idx(text)
    return array

def label_binarize(labels, ctx = mx.cpu(0)):
    lab = nd.zeros(len(data), ctx = ctx)
    for i, label in enumerate(labels):
        if label == 'fake':
            lab[i] = 1
    return lab


class LSTM(gluon.Block):
    def __init__(self, vocab_size, num_embed, num_hidden, num_layers, dropout, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        with self.name_scope():
            self.encoder = gluon.nn.Embedding(vocab_size, num_embed)
            self.LSTM1 = gluon.rnn.LSTM(num_embed, num_layers, layout = 'NTC')
            self.fc1 = gluon.nn.Dense(1, activation='sigmoid')
            
    def forward(self, inputs, hidden):
        emb = self.encoder(inputs)
        output, hidden = self.LSTM1(emb, hidden)
        output = self.fc1(output[:,-1])
        return output, hidden
    
    def begin_state(self, *args, **kwargs):
        return self.LSTM1.begin_state(*args, **kwargs)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments for LSTM model')
	parser.add_argument('train', type=str, help = "Train set dir")
	parser.add_argument('outmodel', type=str, help = "Output file for model")
	parser.add_argument('word2vec', type=str, help = "Path to word2vec gz file")
	parser.add_argument('dictFile', type=str, help = "Path to the dictionary file")
	parser.add_argument('SEQ_LENGTH', type = int, help = "Fixed size length to expand or srink text")
	parser.add_argument('EMBEDDING_DIM', type = int, help = "Size of the embedding dimention")
	parser.add_argument('HIDDEN', type = int, help = "Size of the hidden layer")
	parser.add_argument('LAYERS', type = int, help = "Number of hidden layers")
	parser.add_argument('BATCH_SIZE', type = int, help = "Batch size")
	parser.add_argument('EPOCHS', type = int, help = "Number of epochs for training")

	args = parser.parse_args()

	dictFile = args.dictFile
	trainFile = args.train

	SEQ_LENGTH = args.SEQ_LENGTH
	EMBEDDING_DIM = args.EMBEDDING_DIM
	HIDDEN = args.HIDDEN
	LAYERS = args.LAYERS
	BATCH_SIZE = args.BATCH_SIZE
	EPOCHS = args.EPOCHS
	ctx = mx.gpu(0)

	data = pd.read_json(trainFile, lines=True)
	dct = Dictionary.load(dictFile)

	array = data_to_array(data, ctx = ctx)
	labels = label_binarize(data['type'], ctx = ctx)

	net = LSTM(len(dct), EMBEDDING_DIM, HIDDEN, LAYERS, 0.2)
	net.initialize(mx.init.Normal(sigma=1), ctx = ctx)
	trainer = gluon.Trainer(net.collect_params(), 'rmsprop', {'learning_rate': 0.001})
	loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
	hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx = ctx)

	for epochs in range(0, EPOCHS):
	    total_L = 0.0
	    accuracy = []
	    hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx = ctx)
	    nd_iter = NDArrayIter(data={'data':array},
	                          label={'softmax_label':labels},
	                          batch_size=BATCH_SIZE)
	    acc = mx.metric.Accuracy()
	    for batch in nd_iter:
	        with autograd.record():
	            output, hidden = net(batch.data[0], hidden)
	            L = loss(output, batch.label[0])
	            L.backward()
	            acc.update(output, batch.label[0].flatten())
	        trainer.step(BATCH_SIZE)
	        total_L += mx.nd.sum(L).asscalar()
	        
	    print("Loss : {}, Accuracy : {}".format(total_L, acc.get()))

    net.save_parameters(args.outmodel)