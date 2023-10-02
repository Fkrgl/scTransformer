#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20G
#SBATCH --time=300  # Runtime in D-HH:MM
#SBATCH -o /mnt/qb/work/claassen/cxb257/out/%j.out # Standard output - make sure this is not on $HOME
#SBATCH -e /mnt/qb/work/claassen/cxb257/err/%j.err # Standard error - make sure this is not on $HOME

MODEL=$1
DATA=$2
TOKEN=$3
VOCAB=$4
N_SAMPLES=$5
OUT=$6

python simple_sample.py $MODEL $DATA $TOKEN $VOCAB $N_SAMPLES $OUT