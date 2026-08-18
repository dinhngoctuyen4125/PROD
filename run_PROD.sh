#!/bin/bash

# Running script for the copyrighted code unlearning task

lr=5e-6

Model="codellama/CodeLlama-7b-hf"
ModelName="CodeLlama-7b-hf"
ModelPath="codellama/CodeLlama-7b-hf"
DatasetPath="../Data-Collection/data/codellama/D_forget.json"
SaveModelPath="outputs/models/PROD_lr${lr}"

python PROD.py \
--model_name ${Model} \
--model_path ${ModelPath} \
--output_dir ${SaveModelPath} \
--train_data_path ${DatasetPath} \
--alpha 0.0 \
--num_train_epochs 10 \
--learning_rate ${lr} \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32 \
--logging_steps 1 \
--save_total_limit 2 \
--overwrite_output_dir \
--do_train \
--save_strategy no