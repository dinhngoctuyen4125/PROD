# Large Language Model Unlearning for Source Code

This repository contains the code and data for the paper titled **Large Language Model Unlearning for Source Code**.

## 1. Set Up the Environment

Install **Conda 3.10** and create the environment:

```bash
conda create -n prod python=3.12.13
conda activate prod
```

Install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

## 2. Data

Dữ liệu nằm trong `data/codellama/`:
- **`D_forget.json`** — 10,396 entries, mỗi entry chứa code cần unlearn (`function`) và code retain (`retain`)
- **`D_test.json`** — Dữ liệu test để đánh giá

Trên server, dữ liệu gốc ở `../Data-Collection/codellama/D_forget.json` (cùng cấu trúc).

## 3. Run PROD Training (Unlearning)

Train model unlearn deprecated API bằng phương pháp PROD:

```bash
bash run_PROD.sh
```

**Input**: `D_forget.json` → **Output**: `outputs/models/PROD_lr5e-6/PROD_epoch{N}_lr5e-6/`

## 4. Train OOD Detector

Train bộ phát hiện input thuộc phân phối `D_forget` (RoBERTa + LoRA + contrastive loss + One-Class SVM):

```bash
bash train_ood.sh
```

**Input**: `../Data-Collection/codellama/D_forget.json`
- In-distribution: field `"function"` (forget code) — split 80% train / 20% test
- Out-of-distribution: field `"retain"` (retain code)

**Output**: `ood_checkpoints_codellama_{seed}/`
```
ood_checkpoints_codellama_0/
├── codellama_all_ood_codellama_roberta_ocsvm/   # RoBERTa LoRA weights
├── codellama_all_ood_codellama_ocsvm.pkl         # Fitted OCSVM classifier
├── codellama_all_ood_codellama_mean_list_ocsvm.pt
├── codellama_all_ood_codellama_precision_list_ocsvm.pt
├── codellama_all_ood_codellama_fea_list_ocsvm.pt
├── codellama_all_ood_codellama_gmm_w_ocsvm.pkl   # GMM for weight computation
└── codellama_all_ood_codellama_threshold_ocsvm.json
```

## 5. Inference with OOD-Guided Soft Weighting

Chạy inference kết hợp OOD detector để điều chỉnh mức độ unlearn theo từng input:

```bash
bash eval_soft_infer.sh
```

**Công thức tại mỗi linear layer:**
```
output = (1 - w(x)) × W_base · x  +  w(x) × W_PROD · x
```

Trong đó `w(x)` được tính bởi OOD detector:
| w(x) | Ý nghĩa |
|------|---------|
| 0 | Input KHÔNG thuộc D_forget → dùng base model thuần túy |
| 0.3~0.4 | Partial blend giữa base và PROD |
| 1.2 | Input thuộc D_forget → unlearn mạnh |

**Output**: `results_prod_ood_seed{seed}_{model}_{testfile}` (JSON với kết quả eval + soft-weight summary)

## Pipeline tổng thể

```
Step 1: bash run_PROD.sh          →  PROD model (full fine-tuned)
Step 2: bash train_ood.sh         →  OOD checkpoints (RoBERTa + OCSVM + GMM)
Step 3: bash eval_soft_infer.sh   →  Inference kết hợp OOD gating
```

## Project Structure

```
PROD/
├── PROD.py                  # PROD training script
├── run_PROD.sh              # Shell script chạy PROD training
├── train_ood.py             # OOD detector training
├── train_ood.sh             # Shell script chạy OOD training
├── eval_o3.py               # OOD-guided inference (delta-W hooks)
├── eval_soft_infer.sh       # Shell script chạy inference
├── requirements.txt
├── data/
│   └── codellama/
│       ├── D_forget.json    # Forget data (10,396 entries)
│       └── D_test.json      # Test data
└── src/
    ├── ood_model_selector.py    # RoBERTa OOD model (training + inference)
    ├── ood_data.py              # OOD data loading
    ├── ood_utils.py             # OOD utilities
    ├── ood_calculate_log.py     # OOD metrics (AUROC, TNR, etc.)
    └── ...                      # Other source files
```