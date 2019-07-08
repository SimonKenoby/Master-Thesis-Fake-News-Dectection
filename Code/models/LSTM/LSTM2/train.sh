#!/bin/bash

train="/home/simon/Documents/TFE/Data/train.json"
test="/home/simon/Documents/TFE/Data/test.json"
out="test"
word2vec="/home/simon/Documents/TFE/Data/word2vec-google-news-300.gz"
logs="./logs"
dictFile='Dictionary.dct'
SEQ_LENGTH=5
EMBEDDING_DIM=300
HIDDEN=300
LAYERS=1
BATCH_SIZE=128
EPOCHS=25

python mxnet_bi-lstm.py $train $test $out $word2vec $dictFile $SEQ_LENGTH $EMBEDDING_DIM $HIDDEN $LAYERS $BATCH_SIZE $EPOCHS $logs
rm -r logs