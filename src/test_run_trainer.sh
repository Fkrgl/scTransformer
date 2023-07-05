#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/qb/work/claassen/cxb257/out/%j.out # Standard output - make sure this is not on $HOME
#SBATCH -e /mnt/qb/work/claassen/cxb257/err/%j.err # Standard error - make sure this is not on $HOME
date;hostname;id;pwd

#source /usr/bin/conda
#conda activate transformer
#which python

CELL_TYPE=$1
echo $CELL_TYPE

echo 'preprocess..'
python test_preprocess.py $CELL_TYPE
echo 'train'
python test_train.py $CELL_TYPE
echo 'test'
python test_eval.py $CELL_TYPE