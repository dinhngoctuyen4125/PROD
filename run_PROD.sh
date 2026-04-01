#!/bin/bash
#SBATCH --job-name=PROD_Tuyen
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/user09/huytq/PROD/logs/%x_%j.out
#SBATCH --error=/home/user09/huytq/PROD/logs/%x_%j.out

set -euo pipefail

PROJECT_DIR=/home/user09/huytq/PROD
LOG_DIR=$PROJECT_DIR/logs
HF_CACHE_DIR=/tmp/user09_hf_cache_prod

mkdir -p "$LOG_DIR"
mkdir -p "$HF_CACHE_DIR"

cd "$PROJECT_DIR" || exit 1

source /home/user09/miniconda3/etc/profile.d/conda.sh
conda activate prod_env

export PYTHONNOUSERSITE=1
hash -r

# Hugging Face cache
export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"

export HF_TOKEN="*******"

# Optional: reduce tokenizer parallelism warnings/noise
export TOKENIZERS_PARALLELISM=false

# GPU env cleanup
unset ROCR_VISIBLE_DEVICES || true
unset HIP_VISIBLE_DEVICES || true
unset CUDA_VISIBLE_DEVICES || true

echo "===== START RUN ====="
export HYDRA_FULL_ERROR=1

lr=5e-7

Model="deepseek-ai/deepseek-coder-1.3b-base"
ModelName="deepseek-coder-1.3b"
ModelPath="deepseek-ai/deepseek-coder-1.3b-base"
DatasetPath="data/forget_data/deepseek.json"
SaveModelPath="outputs/models/PROD_lr${lr}"

python PROD.py \
  --model_name "${Model}" \
  --model_path "${ModelPath}" \
  --output_dir "${SaveModelPath}" \
  --train_data_path "${DatasetPath}" \
  --alpha 0.0 \
  --num_train_epochs 10 \
  --learning_rate "${lr}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --logging_steps 1 \
  --save_total_limit 2 \
  --do_train \
  --save_strategy no || exit 1

OutputDir="outputs/results/PROD_lr${lr}"
suffix="2026"

mkdir -p "${OutputDir}"

for file in "${SaveModelPath}"/*; do
  [ -e "$file" ] || continue

  filename=$(basename "$file")
  echo "Filename: ${filename}, Path: ${file}"

  python test_forget_quality.py \
    --model_name "${ModelName}" \
    --model_path "${file}" \
    --dataset "${DatasetPath}" \
    --num-samples 1 \
    --temperature 0.0 \
    --output-dir "${OutputDir}/${filename}/forget_quality" \
    --output-file-suffix "${suffix}"

  python test_model_utility.py \
    --model_name "${ModelName}" \
    --model_path "${file}" \
    --dataset "HumanEval" \
    --num-samples 1 \
    --temperature 0.0 \
    --output-dir "${OutputDir}/${filename}/model_utility" \
    --output-file-suffix "${suffix}"

  python evaluate.py \
    --dataset HumanEval \
    --input_path "${OutputDir}/${filename}/model_utility/HumanEval_${ModelName}_temp0.0_toppNone_topkNone_samples1_0shot_${suffix}.jsonl" \
    --truncate \
    --eval_standard
done