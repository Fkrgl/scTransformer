#!/bin/bash

for i in None
do
  sbatch test_run_trainer.sh $i
done