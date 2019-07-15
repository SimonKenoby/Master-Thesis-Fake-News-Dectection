#!/bin/bash

train="/home/simon/Documents/TFE/Data/train.json"
test_dir="/home/simon/Documents/TFE/Data/"
model_dir="checkpoint/"
logs="test.log"
dictFile='Dictionary.dct'
SEQ_LENGTH=10
EMBEDDING_DIM=100
HIDDEN=100
LAYERS=1
BATCH_SIZE=128

test=$(find  $test_dir -name "*split*")

python buildDictionary.py $train $dictFile
python test_model.py --test $test --input $model_dir --dictFile $dictFile --SEQ_LENGTH $SEQ_LENGTH --EMBEDDING_DIM $EMBEDDING_DIM --HIDDEN $HIDDEN --LAYERS $LAYERS --DROPOUT 0.2 --BATCH_SIZE $BATCH_SIZE --log $logs 
