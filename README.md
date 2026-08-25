# Large Language Model Unlearning for Source Code

## 1. Set Up the Environment

```bash
conda create -n prod python=3.10
conda activate prod
pip install -r requirements.txt
```

## 2. PROD Unlearning & Measure Forget Quality

Train the model to unlearn the specified code API:
```bash
sbatch run_PROD.sh
```

After finetuning, measure the forget quality:
```bash
sudo nohup bash run_forget_quality.sh > logs/forget_quality.log 2>&1 &
```

## 3. Train OOD Detector

```bash
sbatch train_ood.sh
```

## 4. Inference with OOD-Guided Soft Weighting

```bash
sudo nohup bash eval_soft_infer.sh > logs/eval_soft_infer.log 2>&1 &
```

## 5. Test Model Utility (HumanEval)

```bash
sudo nohup bash test_model_utility.sh > logs/test_model_utility.log 2>&1 &
```

> [!WARNING]  
> **Lưu ý cho GPU RTX 5090:** Môi trường `prod` mặc định (CUDA 11.8) sẽ bị lỗi `no kernel image` trên card mới. Hãy cài môi trường `prod_eval` riêng biệt bằng các lệnh sau:
> ```bash
> conda create -n prod_eval python=3.10 -y
> conda activate prod_eval
> tail -n +4 requirements.txt | pip install -r /dev/stdin
> pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
> conda install "mkl<2024.0" -c conda-forge -y
> ```
> *(Đừng quên sửa `test_model_utility.sh` trỏ sang `prod_eval` trước khi chạy)*