#!/bin/bash

#SBATCH --job-name=prod+ood
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

BASE_MODEL="codellama/CodeLlama-7b-hf"
TESTPATH="../Data-Collection/codellama/D_forget.json"

export LD_LIBRARY_PATH=/home/ritsu/miniconda3/envs/prod_eval/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

for EPOCH in 4
do
    PROD_MODEL="tummitum/PROD_epoch${EPOCH}_lr5e-06"
    echo ""
    echo "============================================"
    echo "  Running epoch ${EPOCH}: ${PROD_MODEL}"
    echo "============================================"

    for SEED in 0
    do
        /home/ritsu/miniconda3/envs/prod_eval/bin/python eval_o3.py \
            --test_dataset ${TESTPATH} \
            --base_model ${BASE_MODEL} \
            --prod_model ${PROD_MODEL} \
            --seed ${SEED} \
            --ood_type "_all" \
            --ood_weights "../SimNPO/ood_checkpoints_codellama_0/" \
            --ood_base_model "microsoft/codebert-base" \
            --ood_setting_name "codellama"
    done
done