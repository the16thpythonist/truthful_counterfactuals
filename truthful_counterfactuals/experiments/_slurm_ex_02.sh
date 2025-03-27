#!/bin/bash

# EXPERIMENT 01
# This bash script implements the SLURM scheduling of all the individual experiment runs 
# required for Experiment 1.
# This experiment compares the performance of different uncertainty methods for the 
# ClogP task and most importantly for different testing scenarios which means different 
# either IID testing data or OOD testing data (where there are two OOD cases, once with 
# respect to the output value and once with respect to the input structures)

FOLDER_PATH="/media/ssd/Programming/truthful_counterfactuals"

EXPERIMENTS_PATH="${FOLDER_PATH}/truthful_counterfactuals/experiments"
PYTHON_PATH="${FOLDER_PATH}/venv/bin/python"

identifier="ex_02_c"
#identifier="test"
num_test="0.1"
num_val="0.1"
model="gat"
methods=("mve" "ens" "ens_mve")
test_indices_paths=("None" "'${EXPERIMENTS_PATH}/assets/logp_ood_value.json'" "'${EXPERIMENTS_PATH}/assets/logp_ood_struct.json'")
#test_indices_paths=("None")
seeds=("1" "2" "3" "4" "5")
#seeds=("1")

# ~ Sweep

for u in "${methods[@]}"; do
for p in "${test_indices_paths[@]}"; do
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
        --MODEL_TYPE=\"'${model}'\" \\
        --TEST_INDICES_PATH=\"${p}\" \\
        --EPOCHS=\"50\" \\
    "
done
done
done