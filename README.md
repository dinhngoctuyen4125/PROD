# Large Language Model Unlearning for Source Code

## 1. Set Up the Environment

```bash
conda create -n prod python=3.10
conda activate prod
pip install -r requirements.txt
```

## 2. Run PROD Training (Unlearning)

```bash
sbatch run_PROD.sh
```

## 3. Train OOD Detector

```bash
sbatch train_ood.sh
```

## 4. Inference with OOD-Guided Soft Weighting

```bash
sbatch eval_soft_infer.s
```

## 5. Test Model Utility (HumanEval)

```bash
sudo nohup bash test_model_utility.sh > logs/test_model_utility.log 2>&1 &
```

### Lưu ý: Fix lỗi CUDA cho GPU thế hệ mới (RTX 5090, sm_120)

File `requirements.txt` mặc định cài PyTorch với CUDA 11.8 (`torch==2.7.1+cu118`), chỉ hỗ trợ đến kiến trúc `sm_90`.
Nếu bạn dùng **RTX 5090** (hoặc GPU có CUDA capability `sm_120` trở lên), cần nâng cấp PyTorch thủ công:

**Bước 1: Gỡ PyTorch cũ**
```bash
conda activate prod
pip uninstall -y torch torchvision
```

**Bước 2: Cài PyTorch với CUDA 12.4+**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Bước 3: Xóa file kết quả lỗi từ lần chạy trước (nếu có)**

Do cơ chế resume, script sẽ bỏ qua các task đã có trong file output.
Nếu lần chạy trước bị lỗi CUDA, file output có thể chứa dữ liệu rỗng/sai — cần xóa đi trước khi chạy lại:
```bash
rm -f outputs/results/model_utility/HumanEval_*_2026.jsonl
rm -f outputs/results/model_utility/HumanEval_*_2026.csv
```

**Bước 4: Chạy lại thực nghiệm**
```bash
sudo nohup bash test_model_utility.sh > logs/test_model_utility.log 2>&1 &
```