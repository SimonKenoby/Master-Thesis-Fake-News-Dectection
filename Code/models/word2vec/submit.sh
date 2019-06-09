#!/bin/bash
#
#SBATCH --job-name=keras_gpu_test
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --mem-per-cpu=10000
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:2

SCRATCH=$LOCALSCRATCH/$SLURM_JOB_ID

module load TensorFlow/1.12.0-fosscuda-2018b-Python-3.6.6

srun rm -rf $SCRATCH || exit $?

srun mkdir -p $SCRATCH || exit $?

srun cp -r $HOME/TFE/Code/models/word2vec $SCRATCH || exit $?

srun ls $SCRATCH || exit $?

srun python3 $SCRATCH/word2vec/keras_rnn.py || exit $?

srun mkdir -p $HOME/TFE/Code/models/word2vec/$SLURM_JOB_ID || exit $? 

srun cp $SCRATCH/word2vec/lstm.h5 $HOME/TFE/Code/models/word2vec/$SLURM_JOB_ID || exit $?

srun rm -rf $SCRATCH || exit $?