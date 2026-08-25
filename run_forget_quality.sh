#!/bin/bash

#SBATCH --job-name=forget_quality
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

PROD_MODEL="tummitum/PROD_epoch0_lr5e-06"
INPUT_FILE="../Data-Collection/codellama/D_forget.json"

export LD_LIBRARY_PATH=/home/ritsu/miniconda3/envs/prod_eval/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

for EPOCH in 1 2 3 4 0
do
    PROD_MODEL="tummitum/PROD_epoch${EPOCH}_lr5e-06"
    echo ""
    echo "============================================"
    echo "  Running epoch ${EPOCH}: ${PROD_MODEL}"
    echo "============================================"

    /home/ritsu/miniconda3/envs/prod_eval/bin/python forget_quality.py \
        --model_path ${PROD_MODEL} \
        --input_file ${INPUT_FILE} \
        --batch_size 64 \
        --max_new_tokens 128 \
        --temperature 0.0
done
