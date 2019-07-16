#!/bin/bash

train="/home/simon/Documents/TFE/Data/train2.json"
test_dir="/home/simon/Documents/TFE/Data/"
out_dir="checkpoint/"
out_prefix="model"
utils="/home/simon/Documents/TFE/Code/utils"
logs="train.log"
dictFile='Dictionary.dct'
SEQ_LENGTH_A=(2 3 5 7 10 15 20 25 50)
EMBEDDING_DIM_A=(5 10 20 25 50 100 150 200 300 500)
HIDDEN_A=(2 3 5 7 10 15 25 50 100 200 300 500)
LAYERS_A=(1 2 5 10)
DROPOUT_A=(0 0.2 0.4 0.6 0.8)
BATCH_SIZE=128
EPOCHS=500

: 'SEQ_LENGTH_A=(3 5 7)
EMBEDDING_DIM_A=(5 10)
HIDDEN_A=(2 3)
LAYERS_A=(1)
DROPOUT_A=(0 0.2)
BATCH_SIZE=128
EPOCHS=5'

test=$test_dir/test2.json
#test=$(find . -name "*split*")

for SEQ_LENGTH in ${SEQ_LENGTH_A[@]}; do
	for EMBEDDING_DIM in ${EMBEDDING_DIM_A[@]}; do
		for HIDDEN in ${HIDDEN_A[@]}; do
			for LAYERS in ${LAYERS_A[@]}; do
				for DROPOUT in ${DROPOUT_A[@]}; do
					rm -r $out_dir
					mkdir $out_dir
					python $utils/experiment_number.py --db TFE --collection test --host localhost --port 27017
					experiment=$?
					python buildDictionary.py $train $dictFile
					python attention_lstm.py $train $out_dir/$out_prefix $dictFile $SEQ_LENGTH $EMBEDDING_DIM $HIDDEN $LAYERS $DROPOUT $BATCH_SIZE $EPOCHS $logs --test $test --experiment $experiment --utils $utils --db TFE --collection test --host localhost --port 27017
					python test_model.py --test $test --input $out_dir --dictFile $dictFile --SEQ_LENGTH $SEQ_LENGTH --EMBEDDING_DIM $EMBEDDING_DIM --HIDDEN $HIDDEN --LAYERS $LAYERS --DROPOUT $DROPOUT --BATCH_SIZE $BATCH_SIZE --log $logs --experiment $experiment --utils $utils --db TFE --collection test --host localhost --port 27017
				done
			done
		done
	done
done