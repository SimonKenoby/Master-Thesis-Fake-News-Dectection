#!/bin/bash

train="/home/simon/Documents/TFE/Data/train.json"
out="test"
word2vec="/home/simon/Documents/TFE/Data/word2vec-google-news-300.gz"
dictFile='Dictionary.dct'
SEQ_LENGTH=5
EMBEDDING_DIM=300
HIDDEN=300
LAYERS=1
BATCH_SIZE=128
EPOCHS=250

python buildDictionary.py $train $dictFile
python mxnet_bi-lstm.py $train $out $word2vec $dictFile $SEQ_LENGTH $EMBEDDING_DIM $HIDDEN $LAYERS $BATCH_SIZE $EPOCHS
rm $dictFile