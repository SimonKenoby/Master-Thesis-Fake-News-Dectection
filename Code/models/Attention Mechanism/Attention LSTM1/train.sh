#!/bin/bash

train="/home/simon/Documents/TFE/Data/train.json"
test_dir="/home/simon/Documents/TFE/Data/"
out="test"
word2vec="/home/simon/Documents/TFE/Data/word2vec-google-news-300.gz"
logs="./logs"
dictFile='Dictionary.dct'
SEQ_LENGTH=5
EMBEDDING_DIM=300
HIDDEN=300
LAYERS=1
BATCH_SIZE=128
EPOCHS=10

test="$test_dir/test_splitaa"
python attention_lstm.py $train $out $word2vec $dictFile $SEQ_LENGTH $EMBEDDING_DIM $HIDDEN $LAYERS $BATCH_SIZE $EPOCHS $logs --test $test
