#!/bin/bash

for i in None Alpha Ductal 'Ngn3 high EP' earlystates endstates multiple
do
  sbatch test_run_trainer.sh $i
done