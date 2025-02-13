#!/bin/bash

train="/home/simon/Documents/TFE/Data/train.json"
test_dir="/home/simon/Documents/TFE/Data/"
out_dir="checkpoint/"
out_prefix="model"
word2vec="/home/simon/Documents/TFE/Data/word2vec-google-news-300.gz"
logs="train.log"
dictFile='Dictionary.dct'
SEQ_LENGTH=10
EMBEDDING_DIM=100
HIDDEN=100
LAYERS=1
BATCH_SIZE=128
EPOCHS=25

test="$test_dir/test_splitaa"

rm -r $out_dir
mkdir $out_dir
python buildDictionary.py $train $dictFile
python attention_lstm.py $train $out_dir/$out_prefix $word2vec $dictFile $SEQ_LENGTH $EMBEDDING_DIM $HIDDEN $LAYERS 0.2 $BATCH_SIZE $EPOCHS $logs --test $test
