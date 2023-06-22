#!/bin/bash

WANDB_PROJECT=$1
SWEEP_ID=$2
START_AGENT=/home/claassen/cxb257/scTransformer/src/start_agent.sh

wandb login c697f25b0981fe76f7062d1c3fec4872f9f9c469

for i in {1..4}
do
  sbatch $START_AGENT $WANDB_PROJECT $SWEEP_ID
done