#!/bin/bash

train="/home/simon/Documents/TFE/Data/train2.json"
test_dir="/home/simon/Documents/TFE/Data/"
out_dir="checkpoint/"
out_prefix="model"
word2vec="/home/simon/Documents/TFE/Data/word2vec-google-news-300.gz"
utils="/home/simon/Documents/TFE/Code/utils"
logs="train.log"
dictFile='Dictionary.dct'
SEQ_LENGTH=2
EMBEDDING_DIM=5
HIDDEN=10
LAYERS=1
DROPOUT=0.1
BATCH_SIZE=128
EPOCHS=500

test="$test_dir/test_splitaa"

rm -r $out_dir
mkdir $out_dir
python $utils/experiment_number.py --db TFE --collection test --host localhost --port 27017
experiment=$?
python buildDictionary.py $train $dictFile
python attention_lstm.py $train $out_dir/$out_prefix $dictFile $SEQ_LENGTH $EMBEDDING_DIM $HIDDEN $LAYERS $DROPOUT $BATCH_SIZE $EPOCHS $logs --test $test --experiment 127 --utils $utils --db TFE --collection test --host localhost --port 27017
