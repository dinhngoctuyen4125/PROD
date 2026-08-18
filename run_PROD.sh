#!/bin/bash

#SBATCH --job-name=prod
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

lr=5e-6

Model="codellama/CodeLlama-7b-hf"
ModelName="CodeLlama-7b-hf"
ModelPath="codellama/CodeLlama-7b-hf"
DatasetPath="../Data-Collection/codellama/D_forget.json"
SaveModelPath="outputs/models/PROD_lr${lr}"

python PROD.py \
--model_name ${Model} \
--model_path ${ModelPath} \
--output_dir ${SaveModelPath} \
--train_data_path ${DatasetPath} \
--alpha 0.0 \
--num_train_epochs 5 \
--learning_rate ${lr} \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32 \
--logging_steps 1 \
--save_total_limit 2 \
--overwrite_output_dir \
--do_train \
--save_strategy no