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

echo "===== ENV CHECK ====="
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
which python
python --version
which pip
python -c "import sys; print(sys.executable)"

echo "===== GPU CHECK ====="
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
which nvidia-smi || true
nvidia-smi || true
nvidia-smi -L || true

python - <<'PY'
import os
import sys
import torch

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("SLURM_GPUS_ON_NODE:", os.environ.get("SLURM_GPUS_ON_NODE"))
print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))

try:
    if torch.cuda.device_count() > 0:
        print("device name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("get_device_name error:", repr(e))

try:
    x = torch.tensor([1.0]).to("cuda")
    print("cuda tensor ok:", x)
except Exception as e:
    print("cuda tensor failed:", repr(e))
    sys.exit(1)
PY

echo "===== HF CHECK ====="
python - <<'PY'
import os
print("HF_HOME:", os.environ.get("HF_HOME"))
print("HUGGINGFACE_HUB_CACHE:", os.environ.get("HUGGINGFACE_HUB_CACHE"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
PY

echo "===== START RUN ====="
cd "$PROJECT_DIR" || exit 1

export HYDRA_FULL_ERROR=1

lr=5e-6

# Model="google/codegemma-2b"
# ModelName="codegemma-2b"
# ModelPath="google/codegemma-2b"
# DatasetPath="data/forget_data/codegemma.json"
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