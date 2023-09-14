#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=150G
#SBATCH --time=300  # Runtime in D-HH:MM
#SBATCH -o /mnt/qb/work/claassen/cxb257/out/%j.out # Standard output - make sure this is not on $HOME
#SBATCH -e /mnt/qb/work/claassen/cxb257/err/%j.err # Standard error - make sure this is not on $HOME

python generate_samples.py /mnt/qb/work/claassen/cxb257/models/spleen_t500_ep1.pth /mnt/qb/work/claassen/cxb257/data/cellxgene/spleen.h5ad