#!/bin/bash

train="/home/simon/Documents/TFE/Data/train2.json"
test_dir="/home/simon/Documents/TFE/Data/"
model_dir="checkpoint/"
utils="/home/simon/Documents/TFE/Code/utils"
logs="test.log"
dictFile='Dictionary.dct'
SEQ_LENGTH=20
EMBEDDING_DIM=20
HIDDEN=20
LAYERS=1
DROPOUT=0.8
BATCH_SIZE=128

test=$test_dir/test2.json
#test=$(find . -name "*split*")

python buildDictionary.py $train $dictFile
python $utils/experiment_number.py --db TFE --collection test --host localhost --port 27017
experiment=$?
experiment=$((experiment-1))
python test_model.py --test $test --input $model_dir --dictFile $dictFile --SEQ_LENGTH $SEQ_LENGTH --EMBEDDING_DIM $EMBEDDING_DIM --HIDDEN $HIDDEN --LAYERS $LAYERS --DROPOUT $DROPOUT --BATCH_SIZE $BATCH_SIZE --log $logs --experiment $experiment --utils $utils --db TFE --collection test --host localhost --port 27017
