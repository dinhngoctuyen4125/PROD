#!/bin/bash

#SBATCH --job-name=humaneval
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

MODEL="tummitum/PROD_epoch0_lr5e-06"
MODEL_NAME="PROD_epoch0_lr5e-06"
OUTPUT_DIR="outputs/results/model_utility"
SUFFIX="2026"

/home/ritsu/miniconda3/envs/prod/bin/python test_model_utility.py \
    --model_name ${MODEL} \
    --model_path ${MODEL} \
    --dataset "HumanEval" \
    --num-samples 5 \
    --acctual-num-samples 5 \
    --temperature 0.2 \
    --output-dir ${OUTPUT_DIR} \
    --output-file-suffix ${SUFFIX}

/home/ritsu/miniconda3/envs/prod/bin/python evaluatre.py \
    --dataset HumanEval \
    --input_path "${OUTPUT_DIR}/HumanEval_${MODEL_NAME}_temp0.2_toppNone_topkNone_samples5_0shot_${SUFFIX}.jsonl" \
    --truncate \
    --eval_standard \
    --k_list 1 3 5