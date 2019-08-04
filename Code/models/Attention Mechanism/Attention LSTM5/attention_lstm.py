import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import gluonnlp
from mxnet.io import NDArrayIter
from mxnet import autograd
from mxnet.gluon.data import DataLoader
from multiprocessing import cpu_count

from sklearn.metrics import recall_score, confusion_matrix

from tqdm import tqdm

import json
import argparse
import sys
import warnings

from train import train

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

    warnings.filterwarnings("ignore")

    trainFile = args.train

    SEQ_LENGTH = args.SEQ_LENGTH
    EMBEDDING_DIM = args.EMBEDDING_DIM
    HIDDEN = args.HIDDEN
    LAYERS = args.LAYERS
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS = args.EPOCHS
    DROPOUT = args.DROPOUT
    CPU_COUNT = cpu_count()
    ctx = mx.gpu()

    #mx.random.seed(42, ctx = mx.cpu(0))
    #mx.random.seed(42, ctx = mx.gpu(0))

    sys.path.append(args.utils)

    from register_experiment import Register

    r = Register(args.host, args.port, args.db, args.collection)
    r.newExperiment(r.getLastExperiment() + 1, 'Attention LSTM 5')

    ds = train('train', SEQ_LENGTH, EMBEDDING_DIM, '/home/simon/Documents/TFE/Code/utils')

    net = LSTM(SEQ_LENGTH, EMBEDDING_DIM, HIDDEN, LAYERS, DROPOUT)
    net.initialize(mx.init.Normal(sigma=0.01), ctx = ctx)
    schedule = mx.lr_scheduler.PolyScheduler(max_update=(len(ds) // BATCH_SIZE) * EPOCHS, base_lr=0.01, pwr=2)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.00001, 'wd' : 0.0001, 'lr_scheduler' : schedule})
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx = mx.cpu(0))



    for epochs in range(0, EPOCHS):
        total_L = 0.0
        accuracy = []
        dl = DataLoader(ds, batch_size = BATCH_SIZE, last_batch = 'discard', num_workers=CPU_COUNT // 2)
        hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx = ctx)
        recall_list = []
        cfMatrix = []
        acc = mx.metric.Accuracy()
        pbar = tqdm(total = len(dl))
        for batch, labels in dl:
            pbar.update(1)
            with autograd.record():
                output, hidden = net(batch.copyto(ctx), hidden)
                pred = output > 0.5
                L = loss(output, labels.copyto(ctx))
                L.backward()
                acc.update(labels.flatten(), pred)
                y = labels
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