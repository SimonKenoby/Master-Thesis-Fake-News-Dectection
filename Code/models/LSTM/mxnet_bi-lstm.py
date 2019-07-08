import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import gluonnlp
from mxnet.io import NDArrayIter
from mxnet import autograd

import json
import argparse
from gensim.corpora import Dictionary

def load_data(trainFile, ctx = mx.cpu(0)):
    embed = gluonnlp.embedding.create(embedding_name='word2vec', source="GoogleNews-vectors-negative300")
    labels = []
    num_lines = sum(1 for line in open(trainFile))
    array = nd.ones((num_lines, SEQ_LENGTH, EMBEDDING_DIM), dtype='float32', ctx = ctx)
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
    return array, label_binarize(labels, ctx)

def label_binarize(labels, ctx = mx.cpu(0)):
    lab = nd.zeros(len(labels), ctx = ctx)
    for i, label in enumerate(labels):
        if label == 'fake':
            lab[i] = 1
    return lab


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

	array, labels = load_data(trainFile)

	net = LSTM(len(array), EMBEDDING_DIM, HIDDEN, LAYERS, 0.2)
	net.initialize(mx.init.Normal(sigma=1), ctx = ctx)
	trainer = gluon.Trainer(net.collect_params(), 'rmsprop', {'learning_rate': 0.01})
	loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
	hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx = mx.cpu(0))

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
	            output, hidden = net(batch.data[0].copyto(ctx), hidden)
	            L = loss(output, batch.label[0].copyto(ctx))
	            L.backward()
	            acc.update(output, batch.label[0].flatten())
	        trainer.step(BATCH_SIZE)
	        total_L += mx.nd.sum(L).asscalar()
	        
	    print("Loss : {}, Accuracy : {}".format(total_L, acc.get()))

	net.save_parameters(args.outmodel)