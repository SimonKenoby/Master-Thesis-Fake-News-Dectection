#!/bin/bash
#
#SBATCH --job-name=LSTM_GPU
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=30:00
#SBATCH --mem-per-cpu=20000
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

SCRATCH=$LOCALSCRATCH/$SLURM_JOB_ID

module load Python/3.6.6-fosscuda-2018b
module load OpenMPI/2.1.1-GCC-6.4.0-2.28

SEQ_LENGTH=5
EMBEDDING_DIM=300
HIDDEN=300
LAYERS=1
BATCH_SIZE=128
EPOCHS=1000

rm -rf $SCRATCH || exit $?
mkdir -p $SCRATCH || exit $?

cp -r $HOME/TFE/Code/models/LSTM $SCRATCH || exit $?
cp -r $HOME/TFE/Data/train2.json $SCRATCH/LSTM || exit $?
cp -r $HOME/TFE/Data/word2vec-google-news-300.gz $SCRATCH/LSTM || exit $?

TRAIN=$SCRATCH/LSTM/train2.json
WORD2VEC=$SCRATCH/LSTM/word2vec-google-news-300.gz
OUTPUT=$SCRATCH/LSTM/net.params
dictFile='$SCRATCH/LSTM/Dictionary.dct'


srun python3 buildDictionary.py $train $dictFile
mpirun python3 $SCRATCH/LSTM/mxnet_lstm.py $TRAIN $OUTPUT $WORD2VEC $dictFile $SEQ_LENGTH $EMBEDDING_DIM $HIDDEN $LAYERS $BATCH_SIZE $EPOCHS
mkdir -p $HOME/TFE/output/$SLURM_JOB_ID || exit $? 

cp $OUTPUT $HOME/TFE/output/$SLURM_JOB_ID || exit $?

rm -rf $SCRATCH || exit $?