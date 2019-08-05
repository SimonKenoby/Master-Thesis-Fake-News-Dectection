import warnings

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
from mxnet import autograd
from Attention1 import LSTM
from dataset1 import dataset1
from utils.register_experiment import Register
from mxnet.gluon.data import DataLoader
from multiprocessing import cpu_count

from tqdm import tqdm

import json
import argparse
import numpy as np
from sklearn.metrics import recall_score, confusion_matrix



def recall(y, y_hat):
    y = y.asnumpy()
    y_hat = y_hat.asnumpy()
    return recall_score(y, y_hat), confusion_matrix(y, y_hat).ravel()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Attentionmodel')
    parser.add_argument('--dataset', type=str, help = "Dataset to use for the training")
    parser.add_argument('--filters', type=str, help = "Filters to apply for selecting element in database")
    parser.add_argument('--saveDir', type=str, help = "Directory for checkpoint saving")
    parser.add_argument('--dct', type=str, help = "File for the dictionary")
    parser.add_argument('--SEQ_LENGTH', type=int, help="Fixed size length to expand or srink text")
    parser.add_argument('--EMBEDDING_DIM', type=int, help="Size of the embedding dimention")
    parser.add_argument('--HIDDEN', type=int, help="Size of the hidden layer")
    parser.add_argument('--LAYERS', type=int, help="Number of hidden layers")
    parser.add_argument('--DROPOUT', type=float, help="Dropout value")
    parser.add_argument('--BATCH_SIZE', type=int, help="Batch size")
    parser.add_argument('--EPOCHS', type=int, help="Number of epochs for training")
    parser.add_argument('--utils', type=str, help = "Helper directory")
    parser.add_argument('--db', type=str, help = "DB name", required=True)
    parser.add_argument('--collection', type=str, help = "DB collection")
    parser.add_argument('--host', type=str, help = "DB host")
    parser.add_argument('--port', type=int, help = "Port number of db")

    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    SEQ_LENGTH = args.SEQ_LENGTH
    EMBEDDING_DIM = args.EMBEDDING_DIM
    HIDDEN = args.HIDDEN
    LAYERS = args.LAYERS
    DROPOUT = args.DROPOUT
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS = args.EPOCHS
    CPU_COUNT = cpu_count()
    ctx = mx.gpu()


    ds = dataset1(args.dataset, json.loads(args.filters), SEQ_LENGTH, args.dct)

    r = Register(args.host, args.port, args.db, args.collection)
    r.newExperiment(r.getLastExperiment() + 1, 'Attention')

    net = LSTM(ds.dctLen(), SEQ_LENGTH, EMBEDDING_DIM, HIDDEN, LAYERS, DROPOUT)
    net.initialize(mx.init.Normal(sigma=0.01), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.00001, 'wd': 0.0001})
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx=mx.cpu(0))

    for epochs in range(0, EPOCHS):
        pbar = tqdm(total=len(ds) // BATCH_SIZE)
        total_L = 0.0
        accuracy = []
        hidden = net.begin_state(func=mx.nd.zeros, batch_size=BATCH_SIZE, ctx=ctx)
        dl = DataLoader(ds, batch_size = BATCH_SIZE, last_batch = 'rollover', num_workers=CPU_COUNT // 2)
        recall_list = []
        cfMatrix = []
        acc = mx.metric.Accuracy()
        for batch, labels in dl:
            pbar.update(1)
            with autograd.record():
                output, _ = net(batch.copyto(ctx), hidden)
                L = loss(output, labels.copyto(ctx))
                L.backward()
                pred = output.argmax(axis=1)
                y = labels.argmax(axis=1)
                acc.update(y, pred)
                rec, mat = recall(y, pred)
                recall_list.append(rec)
                cfMatrix.append(mat)
            trainer.step(BATCH_SIZE)
            total_L += mx.nd.sum(L).asscalar()
        pbar.close()
        print("epoch : {}, Loss : {}, Accuracy : {}, recall : {}".format(epochs, total_L, acc.get()[1],
                                                                         np.mean(recall_list)))
        r.addResult({'epoch': epochs,
                     'train': {'accuracy': acc.get()[1], 'loss': total_L, 'recall': np.mean(recall_list),
                               'Confusion Matrix': list(map(int, sum(cfMatrix)))}}, r.getLastExperiment() + 1)
        net.save_parameters(args.saveDir + "_{:04d}.params".format(epochs))
    r.addParams({'SEQ_LENGTH': SEQ_LENGTH,
                 'EMBEDDING_DIM': EMBEDDING_DIM,
                 'HIDDEN': HIDDEN,
                 'LAYERS': LAYERS,
                 'DROPOUT': DROPOUT,
                 'BATCH_SIZE': BATCH_SIZE,
                 'EPOCHS': EPOCHS,
                 'dataset' : args.dataset}, r.getLastExperiment() + 1)
