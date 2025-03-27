#!/bin/bash

# EXPERIMENT 01
# This bash script implements the SLURM scheduling of all the individual experiment runs 
# required for Experiment 1.
# This experiment compares just different configurations of the combined ensemble and mve 
# approach. More specifically different aggregations of the individual uncertainty 
# estimates.

FOLDER_PATH="/media/ssd/Programming/truthful_counterfactuals"

EXPERIMENTS_PATH="${FOLDER_PATH}/truthful_counterfactuals/experiments"
PYTHON_PATH="${FOLDER_PATH}/venv/bin/python"

identifier="ex_05_a"
#identifier="ex_05_test"
num_test="0.1"
num_val="0.1"
model="gat"
method="ens_mve"
aggregations=('mean' 'max' 'min')
seeds=("1" "2" "3" "4" "5")
#seeds=("1")

# ~ Sweep

for s in "${seeds[@]}"; do
for a in "${aggregations[@]}"; do
sbatch \
    --job-name=ex_05 \
    --mem=90GB \
    --time=03:00:00 \
    --wrap="${PYTHON_PATH} ${EXPERIMENTS_PATH}/quantify_uncertainty__${method}.py \\
        --__DEBUG__=\"False\" \\
        --__PREFIX__=\"'${identifier}'\" \\
        --SEED=\"${s}\" \\
        --IDENTIFIER=\"'${identifier}'\" \\
        --NUM_TEST=\"${num_test}\" \\
        --NUM_VAL=\"${num_val}\" \\
        --CALIBRATE_UNCERTAINTY=\"True\" \\
        --MODEL_TYPE=\"'${model}'\" \\
        --TEST_INDICES_PATH=\"None\" \\
        --UNCERTAINTY_AGGREGATION=\"'${a}'\" \\
        --EPOCHS=\"50\" \\
    "
done
done