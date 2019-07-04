#!/bin/bash
#
#SBATCH --job-name=LSTM_GPU
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=30:00
#SBATCH --mem-per-cpu=10000
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

SCRATCH=$LOCALSCRATCH/$SLURM_JOB_ID

module load TensorFlow/1.12.0-fosscuda-2018b-Python-3.6.6
module load OpenMPI/2.1.1-GCC-6.4.0-2.28

srun rm -rf $SCRATCH || exit $?
srun mkdir -p $SCRATCH || exit $?

srun cp -r $HOME/TFE/Code/models/LSTM $SCRATCH || exit $?
srun cp -r $HOME/TFE/Data/train.json $SCRATCH/LSTM || exit $?
srun cp -r $HOME/TFE/Data/word2vec-google-news-300.gz $SCRATCH/LSTM || exit $?

TRAIN=$SCRATCH/LSTM/train.json
WORD2VEC=$SCRATCH/LSTM/word2vec-google-news-300.gz
OUTPUT=$SCRATCH/LSTM/model.h5

mpirun python3 $SCRATCH/LSTM/lstm_word2vec_fakecorpus.py $TRAIN $OUTPUT $WORD2VEC
srun mkdir -p $HOME/TFE/output/$SLURM_JOB_ID || exit $? 

srun cp $SCRATCH/LSTM/model.h5 $HOME/TFE/output/$SLURM_JOB_ID || exit $?

srun rm -rf $SCRATCH || exit $?