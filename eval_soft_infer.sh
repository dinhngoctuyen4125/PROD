BASE_MODEL="codellama/CodeLlama-7b-Instruct-hf"
OOD_SETTING="C"
for SCALE in 0.1
do
  for SEED in 0
  do
    OUTPUT_1="./SCALE_${SCALE}_seed_${SEED}_o_unlearn_lora_checkpoints/lora_forget"

    # Eval on D_test.json
    TESTPATH_1="./data/codellama/D_test.json"
    python eval_o3.py \
      --test_dataset ${TESTPATH_1} \
      --base_model ${BASE_MODEL} \
      --seed ${SEED} \
      --lora_weights ${OUTPUT_1} \
      --ood_type "_all" \
      --ood_setting ${OOD_SETTING} \
      --ood_weights "./ood_checkpoints_codellama_${SEED}/" \
      --ood_base_model "microsoft/codebert-base" \
      --ood_setting_name "codellama"
  done
done