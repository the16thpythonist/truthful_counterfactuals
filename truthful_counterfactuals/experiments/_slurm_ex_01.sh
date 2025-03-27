#!/bin/bash

# EXPERIMENT 01
# This bash script implements the SLURM scheduling of all the individual experiment runs 
# required for Experiment 1.
# This experiment compares the performance of different uncertainty quantification methods 
# on one particular task (ClogP) and for different models.

FOLDER_PATH="/media/ssd/Programming/truthful_counterfactuals"

EXPERIMENTS_PATH="${FOLDER_PATH}/truthful_counterfactuals/experiments"
PYTHON_PATH="${FOLDER_PATH}/venv/bin/python"

identifier="ex_01_a"
#identifier="test"
num_test="0.1"
num_val="0.1"
models=("gin" "gat")
#models=("gin")
methods=("mve" "ens" "swag" "tscore__tan" "tscore__euc" "rep_ens")
methods=("rep_ens")
seeds=("1" "2" "3" "4" "5")
#seeds=("1")

# ~ Sweep

for u in "${methods[@]}"; do
for m in "${models[@]}"; do
for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_01 \
    --mem=90GB \
    --time=03:00:00 \
    --wrap="${PYTHON_PATH} ${EXPERIMENTS_PATH}/quantify_uncertainty__${u}.py \\
        --__DEBUG__=\"False\" \\
        --__PREFIX__=\"'${identifier}'\" \\
        --SEED=\"${s}\" \\
        --IDENTIFIER=\"'${identifier}'\" \\
        --NUM_TEST=\"${num_test}\" \\
        --NUM_VAL=\"${num_val}\" \\
        --CALIBRATE_UNCERTAINTY=\"True\" \\
        --MODEL_TYPE=\"'${m}'\" \\
        --TEST_INDICES_PATH=\"None\" \\
        --EPOCHS=\"50\" \\
    "
done
done
done