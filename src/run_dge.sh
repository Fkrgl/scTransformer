#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --mem=200G
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-02:00  # Runtime in D-HH:MM
#SBATCH --partition=cpu-short
#SBATCH -o /mnt/qb/work/claassen/cxb257/out/%j.out # Standard output - make sure this is not on $HOME
#SBATCH -e /mnt/qb/work/claassen/cxb257/err/%j.err # Standard error - make sure this is not on $HOME

python pyDEseq2.py