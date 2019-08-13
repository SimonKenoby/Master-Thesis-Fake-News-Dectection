#!/bin/bash

train="/home/simon/Documents/TFE/Data/train2.json"
test_dir="/home/simon/Documents/TFE/Data/"
out_dir="checkpoint/"
out_prefix="model"
utils="/home/simon/Documents/TFE/Code/utils"
logs="train.log"
dictFile='Dictionary.dct'
SEQ_LENGTH_A=(20)
EMBEDDING_DIM_A=(10)
HIDDEN_A=(10)
LAYERS_A=(1)
DROPOUT_A=(0.50 0.75)
BATCH_SIZE=512
EPOCHS=1000

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
					python $utils/check_params.py --SEQ_LENGTH $SEQ_LENGTH --EMBEDDING_DIM $EMBEDDING_DIM --HIDDEN $HIDDEN --LAYERS $LAYERS --DROPOUT $DROPOUT --EPOCHS $EPOCHS --Name 'Attention Fake' --db TFE --collection "self_embedding+fake" --host localhost --port 27017;
					check=$?
					if [ $check -eq 0 ]
					then
						rm -r $out_dir
						mkdir $out_dir
						#python $utils/experiment_number.py --db TFE --collection "Attetion3" --host localhost --port 27017
						#experiment=$?
						python buildDictionary.py $train $dictFile
						python attention_lstm.py $train $out_dir/$out_prefix $dictFile $SEQ_LENGTH $EMBEDDING_DIM $HIDDEN $LAYERS $DROPOUT $BATCH_SIZE $EPOCHS $logs --test $test --utils $utils --db TFE --collection "self_embedding+fake" --host localhost --port 27017
						error=$?
						if [ $error -eq 1 ] 
						then
							echo $SEQ_LENGTH
							echo $EMBEDDING_DIM
							echo $HIDDEN
							echo $LAYERS
							echo $DROPOUT
							exit
						fi 
						python test_model.py --test $test --input $out_dir --dictFile $dictFile --SEQ_LENGTH $SEQ_LENGTH --EMBEDDING_DIM $EMBEDDING_DIM --HIDDEN $HIDDEN --LAYERS $LAYERS --DROPOUT $DROPOUT --BATCH_SIZE $BATCH_SIZE --log $logs --utils $utils --db TFE --collection "self_embedding+fake" --host localhost --port 27017
						error=$?
						if [ $error -eq 1 ] 
						then
							echo $SEQ_LENGTH
							echo $EMBEDDING_DIM
							echo $HIDDEN
							echo $LAYERS
							echo $DROPOUT
							exit
						fi 
						sleep 5
					fi
				done
			done
		done
	done
done
