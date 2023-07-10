#!/bin/bash

for i in None Alpha 'Ngn3 high EP' earlystates multiple
do
  sbatch test_run_trainer.sh $i
done