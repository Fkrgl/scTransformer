#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time=30  # Runtime in D-HH:MM
#SBATCH -o /mnt/qb/work/claassen/cxb257/out/%j.out # Standard output - make sure this is not on $HOME
#SBATCH -e /mnt/qb/work/claassen/cxb257/err/%j.err # Standard error - make sure this is not on $HOME

IN_PATH='/mnt/qb/work/claassen/cxb257/data/cellxgene/heart.h5ad'
OUT_PATH=$1
N_SAMPLE=$2

python Sampler.py $IN_PATH $OUT_PATH $N_SAMPLE