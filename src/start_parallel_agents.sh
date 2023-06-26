#!/bin/bash

WANDB_PROJECT=$1
SWEEP_ID=$2
NUM_JOBS=$3
START_AGENT=/home/claassen/cxb257/scTransformer/src/start_agent.sh

wandb login c697f25b0981fe76f7062d1c3fec4872f9f9c469

for i in {1..$NUM_JOBS}
do
  sbatch $START_AGENT $WANDB_PROJECT $SWEEP_ID
done