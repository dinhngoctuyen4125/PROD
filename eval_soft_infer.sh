#!/bin/bash

#SBATCH --job-name=ood
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

BASE_MODEL="codellama/CodeLlama-7b-hf"
PROD_MODEL="outputs/models/PROD_lr5e-6/PROD_epoch0_lr5e-6"

for SEED in 0
do
    # Eval on D_test.json
    TESTPATH="../Data-Collection/codellama/D_test.json"
    python eval_o3.py \
        --test_dataset ${TESTPATH} \
        --base_model ${BASE_MODEL} \
        --prod_model ${PROD_MODEL} \
        --seed ${SEED} \
        --ood_type "_all" \
        --ood_weights "./ood_checkpoints_codellama_${SEED}/" \
        --ood_base_model "microsoft/codebert-base" \
        --ood_setting_name "codellama"
done