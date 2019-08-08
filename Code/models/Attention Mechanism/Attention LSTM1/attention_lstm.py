import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import gluonnlp
from mxnet.io import NDArrayIter
from mxnet import autograd

from sklearn.metrics import recall_score, confusion_matrix

from tqdm import tqdm

import json
import argparse
import sys

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
    return recall_score(y, y_hat), confusion_matrix(y, y_hat).ravel()



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
    def __init__(self, seq_length, num_embed, num_hidden, num_layers, dropout, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.seq_length = seq_length
        with self.name_scope():
            self.LSTM1 = gluon.rnn.LSTM(num_hidden, num_layers, layout = 'NTC', bidirectional = True)
            self.norm1 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)
            self.dropout = gluon.nn.Dropout(dropout)
            self.attention = Attention(seq_length, num_embed, num_hidden, num_layers, dropout)
            self.norm2 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)
            self.fc1 = gluon.nn.Dense(1)
            
    def forward(self, inputs, hidden):
        output, hidden = self.LSTM1(inputs, hidden)
        output = self.norm1(output)
        output = self.dropout(output)
        output = self.attention(output)
        output = self.norm2(output)
        output = self.fc1(output)
        return nd.sigmoid( output ), hidden
    
    def begin_state(self, *args, **kwargs):
        return self.LSTM1.begin_state(*args, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for LSTM model')
    parser.add_argument('--train', type=str, help = "Train set file")
    parser.add_argument('--outmodel', type=str, help = "Output file for model")
    parser.add_argument('--SEQ_LENGTH', type = int, help = "Fixed size length to expand or srink text")
    parser.add_argument('--EMBEDDING_DIM', type = int, help = "Size of the embedding dimention")
    parser.add_argument('--HIDDEN', type = int, help = "Size of the hidden layer")
    parser.add_argument('--LAYERS', type = int, help = "Number of hidden layers")
    parser.add_argument('--DROPOUT', type = float, help = "Dropout value")
    parser.add_argument('--BATCH_SIZE', type = int, help = "Batch size")
    parser.add_argument('--EPOCHS', type = int, help = "Number of epochs for training")
    parser.add_argument('--utils', type=str, help = "Helper directory")
    parser.add_argument('--db', type=str, help = "DB name", required=True)
    parser.add_argument('--collection', type=str, help = "DB collection")
    parser.add_argument('--host', type=str, help = "DB host")
    parser.add_argument('--port', type=int, help = "Port number of db")
    args = parser.parse_args()

    trainFile = args.train

    SEQ_LENGTH = args.SEQ_LENGTH
    EMBEDDING_DIM = args.EMBEDDING_DIM
    HIDDEN = args.HIDDEN
    LAYERS = args.LAYERS
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS = args.EPOCHS
    DROPOUT = args.DROPOUT
    ctx = mx.gpu()

    #mx.random.seed(42, ctx = mx.cpu(0))
    #mx.random.seed(42, ctx = mx.gpu(0))

    sys.path.append(args.utils)

    from register_experiment import Register

    r = Register(args.host, args.port, args.db, args.collection)
    r.newExperiment(r.getLastExperiment() + 1, 'Self_Embedding Attention')

    array, labels = load_data(trainFile)

    net = LSTM(SEQ_LENGTH, EMBEDDING_DIM, HIDDEN, LAYERS, DROPOUT)
    net.initialize(mx.init.Normal(sigma=0.01), ctx = ctx)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001, 'wd' : 0.00001})
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
        recall_list = []
        cfMatrix = []
        acc = mx.metric.Accuracy()
        for batch in nd_iter:
            pbar.update(1)
            with autograd.record():
                output, hidden = net(batch.data[0].copyto(ctx), hidden)
                pred = output > 0.5
                L = loss(output, batch.label[0].copyto(ctx))
                L.backward()
                acc.update(batch.label[0].flatten(), pred)
                y = batch.label[0]
                rec, mat = recall(y, pred)
                recall_list.append(rec)
                cfMatrix.append(mat)
            trainer.step(BATCH_SIZE)
            total_L += mx.nd.sum(L).asscalar()
        pbar.close() 

        # TODO: Evalute on validation set at the same time
        # TODO: Make plots of training 
        print("epoch : {}, Loss : {}, Accuracy : {}, recall : {}".format(epochs, total_L, acc.get()[1], np.mean(recall_list)))
        r.addResult({'epoch' : epochs, 'train' : {'accuracy' : acc.get()[1], 'loss' : total_L, 'recall' : np.mean(recall_list), 'Confusion Matrix' : list(map(int, sum(cfMatrix)))}}, r.getLastExperiment() + 1)
        net.save_parameters(args.outmodel+"_{:04d}.params".format(epochs))
    r.addParams({'SEQ_LENGTH' : SEQ_LENGTH, 'EMBEDDING_DIM': EMBEDDING_DIM, 'HIDDEN': HIDDEN, 'LAYERS' : LAYERS, 'DROPOUT' : DROPOUT, 'BATCH_SIZE' : BATCH_SIZE, 'EPOCHS' : EPOCHS}, r.getLastExperiment() + 1)