#!/bin/bash

train="/home/simon/Documents/TFE/Data/train2.json"
test_dir="/home/simon/Documents/TFE/Data/"
out_dir="checkpoint/"
out_prefix="model"
utils="utils/"
logs="train.log"
dictFile='Dictionary.dct'
SEQ_LENGTH_A=(20)
EMBEDDING_DIM_A=(100)
HIDDEN_A=(100)
LAYERS_A=(1)
DROPOUT_A=(0.25 0.50)
BATCH_SIZE=128
EPOCHS=100

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
					python $utils/check_params.py --SEQ_LENGTH $SEQ_LENGTH --EMBEDDING_DIM $EMBEDDING_DIM --HIDDEN $HIDDEN --LAYERS $LAYERS --DROPOUT $DROPOUT --EPOCHS $EPOCHS --Name 'Self_Embedding LSTM 3.2' --db TFE --collection "Attetion3" --host localhost --port 27017;
					check=$?
					if [ $check -eq 0 ]
					then
						rm -r $out_dir
						mkdir $out_dir
						python buildDictionary.py $train $dictFile
						python train.py --dataset liar_liar \
						--filters '{"split" : "train"}' \
						--dct $dictFile \
						--saveDir $out_dir/$out_prefix \
						--SEQ_LENGTH $SEQ_LENGTH \
						--EMBEDDING_DIM $EMBEDDING_DIM \
						--HIDDEN $HIDDEN \
						--LAYERS $LAYERS \
						--DROPOUT $DROPOUT \
						--BATCH_SIZE $BATCH_SIZE \
						--EPOCHS $EPOCHS \
						--utils $utils \
						--db TFE \
						--collection Self_Embedding \
						--host localhost \
						--port 27017
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
						python test_model.py 
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