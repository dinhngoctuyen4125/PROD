lr=5e-7

Model="deepseek-ai/deepseek-coder-1.3b-instruct"
ModelName="deepseek-coder-1.3b"
ModelPath="deepseek-ai/deepseek-coder-1.3b-instruct"
DatasetPath="data/forget_data/deepseek_not_hint.json"
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

  # python test_model_utility.py \
  #   --model_name "${ModelName}" \
  #   --model_path "${file}" \
  #   --dataset "HumanEval" \
  #   --num-samples 1 \
  #   --temperature 0.0 \
  #   --output-dir "${OutputDir}/${filename}/model_utility" \
  #   --output-file-suffix "${suffix}"

  # python evaluate.py \
  #   --dataset HumanEval \
  #   --input_path "${OutputDir}/${filename}/model_utility/HumanEval_${ModelName}_temp0.0_toppNone_topkNone_samples1_0shot_${suffix}.jsonl" \
  #   --truncate \
  #   --eval_standard
done