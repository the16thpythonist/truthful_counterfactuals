#!/bin/bash
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --time=03:00:00
#SBATCH --priority=10000
#SBATCH --job-name=cf_abl

# Making sure to use the correct python env
source /media/ssd/Programming/truthful_counterfactuals/venv/bin/activate
which python

# ~ ens mve
python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/counterfactual_truthfulness__ens_mve.py \
    --__DEBUG__=False \
    --NUM_COUNTERFACTUAL_ORIGINALS=100 \
    --NUM_COUNTERFACTUALS=10 \
    --NUM_VAL=10000 \ 
    --NUM_TEST=800

# ~random
# python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/counterfactual_truthfulness__rand.py \
#     --__DEBUG__=False \
#     --NUM_COUNTERFACTUAL_ORIGINALS=100 \
#     --NUM_COUNTERFACTUALS=10

# ~ aqsoldb
# python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/quantify_uncertainty__ens_mve__aqsoldb.py \
#     --__DEBUG__=False

# ~ compas
# python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/quantify_uncertainty__ens_mve__compas.py \
#     --__DEBUG__=False

# ~ qm9 energy
# python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/quantify_uncertainty__ens_mve__qm9.py \
#     --__DEBUG__=False \
#     --TARGET="'dipole'"

# ~ lipop
# python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/quantify_uncertainty__ens_mve__lipop.py \
#     --__DEBUG__=False

# ~ tadf
# python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/quantify_uncertainty__ens_mve__tadf.py \
#     --__DEBUG__=False

# ~ mve ood struct
# python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/quantify_uncertainty__mve.py \
#     --__DEBUG__=False \
#     --CALIBRATE_UNCERTAINTY=True \
#     --TEST_INDICES_PATH="'/media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/assets/logp_ood_struct.json'"

# ~ ens ood struct
# python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/quantify_uncertainty__ens.py \
#     --__DEBUG__=False \
#     --CALIBRATE_UNCERTAINTY=True \
#     --TEST_INDICES_PATH="'/media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/assets/logp_ood_struct.json'"

# ~ ens_mve ood struct
# python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/quantify_uncertainty__ens_mve.py \
#     --__DEBUG__=False \
#     --CALIBRATE_UNCERTAINTY=True \
#     --TEST_INDICES_PATH="'/media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/assets/logp_ood_struct.json'"

# ~ mve iid
# python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/quantify_uncertainty__mve.py \
#     --__DEBUG__=False \
#     --CALIBRATE_UNCERTAINTY=True \
#     --TEST_INDICES_PATH=None

# ~ ens mve iid
# python /media/ssd/Programming/truthful_counterfactuals/truthful_counterfactuals/experiments/quantify_uncertainty__ens_mve.py \
#     --__DEBUG__=False \
#     --CALIBRATE_UNCERTAINTY=True \
#     --TEST_INDICES_PATH=None