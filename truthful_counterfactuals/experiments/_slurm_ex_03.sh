#!/bin/bash

# EXPERIMENT 03
# This bash script implements the SLURM scheduling of all the individual experiment runs 
# required for Experiment 3.
# This experiment compares the results of the best performing model+method combination 
# (ensemble + MVE) on multiple different datasets. This is to show that the previous 
# results are independent of the dataset and that the method is generally applicable.

FOLDER_PATH="/media/ssd/Programming/truthful_counterfactuals"

EXPERIMENTS_PATH="${FOLDER_PATH}/truthful_counterfactuals/experiments"
PYTHON_PATH="${FOLDER_PATH}/venv/bin/python"

identifier="ex_03_a"
num_test="0.1"
num_val="0.1"
model="gat"
method="ens_mve"
seeds=("1" "2" "3" "4" "5")
#seeds=("1")

# ~ logp solubility

dataset="${EXPERIMENTS_PATH}/assets/logp"

for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_03 \
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
        --EPOCHS=\"50\" \\
        --VISUAL_GRAPH_DATASET=\"'${dataset}'\" \\
    "
done

# ~ aqsoldb solubility

dataset="/media/ssd/.visual_graph_datasets/datasets/aqsoldb"

for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_03 \
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
        --EPOCHS=\"50\" \\
        --VISUAL_GRAPH_DATASET=\"'${dataset}'\" \\
    "
done

# ~ lipop 

dataset="/media/ssd/.visual_graph_datasets/datasets/lipophilicity"

for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_03 \
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
        --EPOCHS=\"50\" \\
        --VISUAL_GRAPH_DATASET=\"'${dataset}'\" \\
    "
done

# ~ COMPAS

num_test="0.2"
dataset="/media/ssd/.visual_graph_datasets/datasets/compas"
# (GAP, E_rel)
targets=("0" "1")

for t in "${targets[@]}"; do
for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_03 \
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
        --EPOCHS=\"50\" \\
        --VISUAL_GRAPH_DATASET=\"'${dataset}'\" \\
        --TARGET_INDEX=\"${t}\" \\
    "
done
done

# ~ QM9

num_test="0.3"
dataset="/media/ssd/.visual_graph_datasets/datasets/qm9"
# (mu, homo, lumo, gap)
targets=("3" "5" "6" "7")

for t in "${targets[@]}"; do
for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_03 \
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
        --EPOCHS=\"50\" \\
        --VISUAL_GRAPH_DATASET=\"'${dataset}'\" \\
        --TARGET_INDEX=\"${t}\" \\
    "
done
done