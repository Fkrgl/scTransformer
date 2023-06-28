#!/bin/bash

WANDB_PROJECT=$1
SWEEP_ID=$2
END=$3
START_AGENT=/home/claassen/cxb257/scTransformer/src/start_agent.sh

wandb login c697f25b0981fe76f7062d1c3fec4872f9f9c469

for (( c=1; c<=$END; c++ ))
do
  sbatch $START_AGENT $WANDB_PROJECT $SWEEP_ID
done