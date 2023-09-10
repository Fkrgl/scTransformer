#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --time=1  # Runtime in D-HH:MM
#SBATCH --partition=cpu-long
#SBATCH -o /mnt/qb/work/claassen/cxb257/out/%j.out # Standard output - make sure this is not on $HOME
#SBATCH -e /mnt/qb/work/claassen/cxb257/err/%j.err # Standard error - make sure this is not on $HOME


python3 download_cellxgene_datasets.py