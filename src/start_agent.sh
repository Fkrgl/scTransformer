#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=1-00:00  # Runtime in D-HH:MM
#SBATCH --gres=gpu:1    # optionally type and number of gpus
#SBATCH --partition=gpu-2080ti
#SBATCH -o /mnt/qb/work/claassen/cxb257/out/%j.out # Standard output - make sure this is not on $HOME
#SBATCH -e /mnt/qb/work/claassen/cxb257/err/%j.err # Standard error - make sure this is not on $HOME

WANDB_PROJECT=$1
SWEEP_ID=$2
WANDB__SERVICE_WAIT=300
#wandb login c697f25b0981fe76f7062d1c3fec4872f9f9c469
wandb agent -p $WANDB_PROJECT $SWEEP_ID
#cd /project_path && PYTHONPATH=/project_path WANDB_ENTITY=entity WANDB_PROJECT=project wandb agent $2
