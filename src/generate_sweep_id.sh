#!/bin/bash
WANDB_PROJECT=$1
CONFIG_YAML=$2

# script prints a sweep id which can be used for the agent in sweep_training.sh
export WANDB__SERVICE_WAIT=300
wandb login c697f25b0981fe76f7062d1c3fec4872f9f9c469
wandb sweep --project $WANDB_PROJECT $CONFIG_YAML