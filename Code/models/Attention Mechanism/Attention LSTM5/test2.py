import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import gluonnlp
from tqdm import tqdm
from mxnet.gluon.data import DataLoader
from multiprocessing import cpu_count


import json
import argparse
import os
import sys
import numpy as np
from sklearn.metrics import recall_score, confusion_matrix
from testset import testset

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


def recall(y, y_hat):
    y = y.asnumpy()
    y_hat = y_hat.asnumpy()
    return recall_score(y, y_hat), confusion_matrix(y, y_hat, labels = [0,1]).ravel()


class Attention(gluon.Block):
    def __init__(self, seq_length, num_embed, num_hidden, num_layers, dropout, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        self.seq_length = seq_length
        with self.name_scope():
            self.fc1 = gluon.nn.Dense(seq_length)
            
    def forward(self, hidden):
        h_f = hidden[:,:,0:self.num_hidden]
        h_b = hidden[:,:,self.num_hidden:2*self.num_hidden]
        H = (h_f + h_b).transpose((0,2,1))
        M = nd.tanh(H)
        alpha = nd.softmax(self.fc1(M)).reshape((0,0,-1))
        r = nd.batch_dot(H, alpha)
        return nd.tanh(r)


class LSTM(gluon.Block):
    def __init__(self, seq_length, num_embed, num_hidden, num_layers, dropout, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.seq_length = seq_length
        with self.name_scope():
            self.LSTM1 = gluon.rnn.LSTM(num_hidden, num_layers, layout = 'NTC', bidirectional = True)
            self.dropout = gluon.nn.Dropout(dropout)
            self.attention = Attention(seq_length, num_embed, num_hidden, num_layers, dropout)
            self.fc1 = gluon.nn.Dense(1)
            
    def forward(self, inputs, hidden):
        output, hidden = self.LSTM1(inputs, hidden)
        output = self.dropout(output)
        output = self.attention(output)
        output = self.fc1(output)
        return nd.sigmoid( output ), hidden
    
    def begin_state(self, *args, **kwargs):
        return self.LSTM1.begin_state(*args, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for LSTM model')
    parser.add_argument('--test', nargs='+', type=str, help = "Validation set file", required=True)
    parser.add_argument('--input', type=str, help = "Input directory for the model files")
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
    CPU_COUNT = cpu_count()
    ctx = mx.gpu()

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(args.input):
        for file in f:
            files.append(os.path.join(r, file))
    files.sort()

    r = Register(args.host, args.port, args.db, args.collection)
    print(r.getLastExperiment())

    ds = testset('train', SEQ_LENGTH, EMBEDDING_DIM, '/home/simon/Documents/TFE/Code/utils')
    acc = mx.metric.Accuracy()
    accuracy = []
    for j, model in enumerate(files):
        recall_list = []
        cfMatrix = []
        net = LSTM(SEQ_LENGTH, EMBEDDING_DIM, HIDDEN, LAYERS, DROPOUT)
        net.load_parameters(model, ctx=ctx)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, last_batch='discard', num_workers=CPU_COUNT)
        pbar = tqdm(total = len(dl))
        for batch, labels in dl:
            hidden = net.begin_state(func=mx.nd.zeros, batch_size=len(batch), ctx = ctx)
            output, _ = net(batch.copyto(ctx), hidden)
            pred = output > 0.5
            y = labels
            acc.update(y, pred)
            rec, mat = recall(y, pred)
            recall_list.append(rec)
            cfMatrix.append(mat)
            pbar.update(1)
        accuracy.append(acc.get()[1])
        r.addEpochs(j, {'accuracy' : acc.get()[1], 'recall' : np.mean(recall_list), 'Confusion Matrix' : list(map(int, sum(cfMatrix)))}, r.getLastExperiment() + 1, 'valid')
        pbar.close()
    r.closeExperiment(r.getLastExperiment() + 1)