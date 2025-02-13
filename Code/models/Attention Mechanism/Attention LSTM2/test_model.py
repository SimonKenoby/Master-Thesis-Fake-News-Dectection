import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
from mxnet.io import NDArrayIter
from tqdm import tqdm

import json
import argparse
import pandas as pd
import os
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
    array = [dct.token2id[token] if token in dct.token2id else -1 for token in tokens]
    if len(array) > SEQ_LENGTH:
        array = array[0:SEQ_LENGTH]
    else:
        array.extend([-1 for i in range(0, SEQ_LENGTH - len(array))])
    return nd.array(array, ctx = ctx, dtype='int32')

def label_binarize(labels, ctx = mx.cpu(0)):
    lab = nd.zeros(len(labels), ctx = ctx)
    for i, label in enumerate(labels):
        if label == 'fake':
            lab[i] = 1
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
            self.fc1 = gluon.nn.Dense(1)
            
    def forward(self, inputs, hidden):
        emb = self.encoder(inputs)
        output, hidden = self.LSTM1(emb, hidden)
        output = self.attention(output)
        output = self.fc1(output)
        return nd.sigmoid( output ), hidden
    
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
    parser.add_argument('--log', type=str, help = "Log Directory")

    args = parser.parse_args()

    dictFile = args.dictFile
    testFiles = args.test

    SEQ_LENGTH = args.SEQ_LENGTH
    EMBEDDING_DIM = args.EMBEDDING_DIM
    HIDDEN = args.HIDDEN
    LAYERS = args.LAYERS
    DROPOUT = args.DROPOUT
    BATCH_SIZE = args.BATCH_SIZE
    LOG = args.log
    ctx = mx.gpu(0)

    cw = csvwriter(LOG)
    cw.write(['accuracy'])

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(args.input):
        for file in f:
            files.append(os.path.join(r, file))
    files.sort()

    dct = Dictionary.load(dictFile)
    pbar = tqdm(len(testFiles))
    for test_file in testFiles:
        array, labels = load_data(test_file, dct)
        acc = mx.metric.Accuracy()
        accuracy = []
        for model in files:
            net = LSTM(len(dct), SEQ_LENGTH, EMBEDDING_DIM, HIDDEN, LAYERS, DROPOUT)
            net.load_parameters(model, ctx=ctx)
            hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx = ctx)
            nd_iter = NDArrayIter(data={'data':array},
                              label={'softmax_label':labels},
                              batch_size=BATCH_SIZE)
            for batch in nd_iter:
                output, _ = net(batch.data[0].copyto(ctx), hidden)
                pred = output > 0.5
                acc.update(batch.label[0].flatten(), pred)
            accuracy.append(acc.get()[1])
        cw.write(accuracy)
        pbar.update(1)
    pbar.close()    
    cw.close()
        