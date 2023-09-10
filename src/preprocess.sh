#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH --time=30  # Runtime in D-HH:MM
#SBATCH -o /mnt/qb/work/claassen/cxb257/out/%j.out # Standard output - make sure this is not on $HOME
#SBATCH -e /mnt/qb/work/claassen/cxb257/err/%j.err # Standard error - make sure this is not on $HOME

IN_PATH=$1
N_HVG=$2
OUT_PATH=$3

python preprocessor.py $IN_PATH $N_HVG -path_out $OUT_PATH