#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/qb/work/claassen/cxb257/out/hostname_%j.out # Standard output - make sure this is not on $HOME
#SBATCH -e /mnt/qb/work/claassen/cxb257/err/hostname_%j.err # Standard error - make sure this is not on $HOME
date;hostname;id;pwd

source /usr/bin/conda
conda activate transformer
which python

config.yaml='/home/claassen/cxb257/scTransformer/config.yaml'
train_file='/home/claassen/cxb257/scTransformer/src/trainer.py'
project_name='cluster_hyperparameter_search_test'

echo 'run script'
python wandb_on_slurm.py $config_yaml $train_file $project_name