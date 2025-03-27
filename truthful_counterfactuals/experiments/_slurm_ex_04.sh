#!/bin/bash

# EXPERIMENT 04
# This bash script implements the SLURM scheduling of all the individual experiment runs 
# required for Experiment 4.
# This experiment compares calculates the truthfulness on the counterfactuals for different 
# methods and different seeds.

FOLDER_PATH="/media/ssd/Programming/truthful_counterfactuals"

EXPERIMENTS_PATH="${FOLDER_PATH}/truthful_counterfactuals/experiments"
PYTHON_PATH="${FOLDER_PATH}/venv/bin/python"

identifier="ex_04_a"
num_test="0.1"
num_val="0.2"
model="gat"
methods=("ens" "mve" "ens_mve")
seeds=("1" "2" "3" "4" "5")

for m in "${methods[@]}"; do
for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_04 \
    --mem=90GB \
    --time=03:00:00 \
    --wrap="${PYTHON_PATH} ${EXPERIMENTS_PATH}/counterfactual_truthfulness__${m}.py \\
        --__DEBUG__=\"False\" \\
        --__PREFIX__=\"'${identifier}'\" \\
        --SEED=\"${s}\" \\
        --IDENTIFIER=\"'${identifier}'\" \\
        --NUM_TEST=\"${num_test}\" \\
        --NUM_VAL=\"${num_val}\" \\
        --CALIBRATE_UNCERTAINTY=\"True\" \\
        --MODEL_TYPE=\"'${model}'\" \\
        --EPOCHS=\"50\" \\
    "
done
done
