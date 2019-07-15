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
import pandas as pd
from gensim.corpora import Dictionary

from csvwriter import csvwriter

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
    array = [dct.token2id[token] for token in tokens]
    if len(array) > SEQ_LENGTH:
        array = array[0:SEQ_LENGTH]
    else:
        array.extend([-1 for i in range(0, SEQ_LENGTH - len(array))])
    return nd.array(array, ctx = ctx,  dtype='int32')

def label_binarize(labels, ctx = mx.cpu(0)):
    lab = nd.zeros((len(labels), 2), ctx = ctx)
    for i, label in enumerate(labels):
        if label == 'fake':
            lab[i, 1] = 1
        else:
            lab[i, 0] = 1
    return lab


class Attention(gluon.Block):
    def __init__(self, seq_length, num_embed, num_hidden, num_layers, dropout, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        self.seq_length = seq_length
        with self.name_scope():
            self.fc1 = gluon.nn.Dense(seq_length, activation = 'relu')
            
    def forward(self, hidden):
        h_f = hidden[:,:,0:self.num_hidden]
        h_b = hidden[:,:,self.num_hidden:2*self.num_hidden]
        H = (h_f + h_b).transpose((0,2,1))
        M = nd.tanh(H)
        alpha = nd.softmax(self.fc1(M)).reshape((0,0,-1))
        r = nd.batch_dot(H, alpha)
        return nd.tanh(r)


class LSTM(gluon.Block):
    def __init__(self, vocab_size, seq_length, num_embed, num_hidden, num_layers, dropout, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.seq_length = seq_length
        with self.name_scope():
            self.encoder = gluon.nn.Embedding(vocab_size, num_embed)
            self.LSTM1 = gluon.rnn.LSTM(num_embed, num_layers, layout = 'NTC', bidirectional = True)
            self.attention = Attention(seq_length, num_embed, num_hidden, num_layers, dropout)
            self.fc1 = gluon.nn.Dense(2)
            
    def forward(self, inputs, hidden):
        emb = self.encoder(inputs)
        output, hidden = self.LSTM1(emb, hidden)
        output = self.attention(output)
        output = self.fc1(output)
        return nd.softmax( output , axis=1), hidden
    
    def begin_state(self, *args, **kwargs):
        return self.LSTM1.begin_state(*args, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for LSTM model')
    parser.add_argument('train', type=str, help = "Train set file")
    parser.add_argument('--test', nargs='+', type=str, help = "Validation set file", required=True)
    parser.add_argument('outmodel', type=str, help = "Output file for model")
    parser.add_argument('word2vec', type=str, help = "Path to word2vec gz file")
    parser.add_argument('dictFile', type=str, help = "Path to the dictionary file")
    parser.add_argument('SEQ_LENGTH', type = int, help = "Fixed size length to expand or srink text")
    parser.add_argument('EMBEDDING_DIM', type = int, help = "Size of the embedding dimention")
    parser.add_argument('HIDDEN', type = int, help = "Size of the hidden layer")
    parser.add_argument('LAYERS', type = int, help = "Number of hidden layers")
    parser.add_argument('DROPOUT', type = float, help = "Dropout value")
    parser.add_argument('BATCH_SIZE', type = int, help = "Batch size")
    parser.add_argument('EPOCHS', type = int, help = "Number of epochs for training")
    parser.add_argument('log', type=str, help = "Log Directory")

    args = parser.parse_args()

    dictFile = args.dictFile
    trainFile = args.train
    testFile = args.test[0]

    SEQ_LENGTH = args.SEQ_LENGTH
    EMBEDDING_DIM = args.EMBEDDING_DIM
    HIDDEN = args.HIDDEN
    LAYERS = args.LAYERS
    DROPOUT = args.DROPOUT
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS = args.EPOCHS
    LOG = args.log
    ctx = mx.gpu(0)

    cw = csvwriter(LOG)
    cw.write(['epoch', 'loss', 'accuracy'])

    #mx.random.seed(42, ctx = mx.cpu(0))
    #mx.random.seed(42, ctx = mx.gpu(0))

    dct = Dictionary.load(dictFile)
    array, labels = load_data(trainFile, dct)

    net = LSTM(len(dct), SEQ_LENGTH, EMBEDDING_DIM, HIDDEN, LAYERS, 0.2)
    net.initialize(mx.init.Normal(sigma=1), ctx = ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05, 'wd' : 0.00001})
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
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
                output, _ = net(batch.data[0].copyto(ctx), hidden)
                L = loss(output, batch.label[0].copyto(ctx))
                L.backward()
                pred = output.argmax(axis=1)
                y = batch.label[0].argmax(axis=1)
                acc.update(y, pred)
            trainer.step(BATCH_SIZE)
            total_L += mx.nd.sum(L).asscalar()
        pbar.close() 

        # TODO: Evalute on validation set at the same time
        # TODO: Make plots of training 
        print("epoch : {}, Loss : {}, Accuracy : {}".format(epochs, total_L, acc.get()[1]))
        cw.write([epochs, total_L, acc.get()[1]])
        net.save_parameters(args.outmodel+"_{:04d}.params".format(epochs))

    cw.close()
