# OOD-Guided Soft-Weighted LoRA Inference — Kiến trúc & Tài liệu kỹ thuật

> Trích xuất từ dự án O3 Unlearning Experiment.  
> Mục đích: Tài liệu tham khảo để tái sử dụng kiến trúc OOD sang dự án khác.

---

## 1. Tổng quan kiến trúc

Hệ thống gồm **2 model riêng biệt** phối hợp tại inference:

```
                         Input x (text)
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
   ┌─────────────────────┐         ┌─────────────────────────┐
   │   OOD Detector      │         │   LLM (CodeLlama)       │
   │   (RoBERTa + LoRA)  │         │   + Unlearn LoRA        │
   │                     │         │                         │
   │  Input → Features   │         │  Mỗi LoRA layer áp dụng │
   │    → Mah Score      │         │  công thức:              │
   │    → OCSVM Score    │         │                         │
   │    → GMM → w(x)     │         │  out = W·x·1 + ΔW·x·w  │
   └────────┬────────────┘         └────────────┬────────────┘
            │                                   │
            │  w(x) ∈ {0, (0.3,0.4], 1.2}      │
            └──────────────►────────────────────┘
                                    │
                                    ▼
                              Generated text
```

**Ý tưởng cốt lõi**: Mỗi input `x` tại inference được OOD detector đánh giá xem có thuộc "forget domain" hay không. Nếu thuộc → LoRA unlearn được kích hoạt mạnh (`w=1.2`). Nếu không thuộc → LoRA bị tắt (`w=0`), model hoạt động như base model gốc.

---

## 2. Module OOD Detector

### 2.1 Kiến trúc model

**Base model**: RoBERTa (ví dụ `microsoft/codebert-base`)  
**Fine-tuning**: LoRA trên `query`, `key`, `value` layers  
**Output**: 13 hidden state representations (1 embedding + 12 transformer layers)

**File**: `src/ood_model_selector.py`

| Class | Mục đích |
|-------|----------|
| `RobertaForSelector` | **Training** — train OOD detector với contrastive loss (InfoNCE-style) |
| `RobertaForSelector_inference` | **Inference** — load pretrained LoRA, tính Mahalanobis score |

### 2.2 Training OOD Detector

**File**: `train_ood.py`  
**Script**: `train_ood.sh`

#### Dữ liệu cần chuẩn bị:
- **In-distribution (ID)**: Dữ liệu thuộc domain cần unlearn (forget data)
  - Split 80/20: 80% train, 20% test
- **Out-of-distribution (OOD)**: Dữ liệu retain (không unlearn)

#### Luồng training:

```
1. Forward RoBERTa + LoRA (query encoder):
   batch_mlm → 13 hidden states → mean pooling → z_1 (per layer)

2. Forward RoBERTa (key encoder, frozen):
   batch → 13 hidden states → mean pooling → z_2 (per layer)

3. Loss = Σ(layer=0..12) mean(entropy(softmax(z_1 · z_2^T)))
   → Contrastive-style loss để học feature representations

4. Sau mỗi epoch, chạy OOD detection evaluation:
   - Tính Mahalanobis score trên train/test/ood sets
   - Fit OCSVM trên train scores
   - Lưu best model theo accuracy
```

#### Artifacts được lưu (sau training):

```
ood_checkpoints_{save_name}_{seed}/
├── {dataset}_{ood_dataset}_roberta_ocsvm/          # RoBERTa LoRA weights
├── {dataset}_{ood_dataset}_ocsvm.pkl                # Fitted OCSVM classifier
├── {dataset}_{ood_dataset}_mean_list_ocsvm.pt       # Per-layer mean vectors (13 tensors)
├── {dataset}_{ood_dataset}_precision_list_ocsvm.pt  # Per-layer precision matrices
├── {dataset}_{ood_dataset}_fea_list_ocsvm.pt        # Per-layer normalized features
├── {dataset}_{ood_dataset}_gmm_w_ocsvm.pkl          # GMM for weight computation
└── {dataset}_{ood_dataset}_threshold_ocsvm.json     # [x0, threshold, accuracy]
```

### 2.3 Feature Extraction & Scoring

#### Bước 1: Extract Mahalanobis features (`sample_X_estimator`)

Chạy trên **toàn bộ training set** để tính thống kê tham chiếu:

```python
for layer_i in range(13):
    features_i = mean_pool(hidden_states[layer_i])   # (N, 768)
    mean_list[i] = mean(features_i)                   # (768,)
    precision_list[i] = EmpiricalCovariance(
        features_i - mean
    ).precision_                                      # (768, 768)
    fea_list[i] = L2_normalize(features_i)            # (N, 768)
```

#### Bước 2: Compute Mahalanobis + Cosine Score (`get_unsup_Mah_score_s`)

Cho mỗi batch input:

```python
for layer_i in range(13):
    feature = mean_pool(hidden_states[layer_i])      # (batch, 768)
    
    # Mahalanobis distance
    zero_f = feature - mean_list[i]
    gaussian_score = -0.5 * (zero_f @ precision[i] @ zero_f.T).diag()
    
    # Cosine similarity (max over training features)
    cs_score = max(normalize(feature) @ fea_list[i].T)
    
    # Combined score
    score[i] = -cs_score * 1000 + gaussian_score
```

Output: `(batch_size, 13)` → slice `[:, 1:]` → `(batch_size, 12)` (bỏ layer 0)

#### Bước 3: OCSVM Score

```python
ocsvm_score = ocsvm.score_samples(mah_score)  # (batch_size,) — scalar per sample
```

### 2.4 Weight Computation (`obtain_weights`)

Từ OCSVM score → soft weight `w(x)`:

```python
def obtain_weights(input_x, gmm, x0):
    # 1. Cumulative probability qua GMM
    cp_x = gmm_cdf(input_x, gmm)
    cp_symmetric = gmm_cdf(2*x0 - input_x, gmm)
    
    # 2. Symmetric probability
    cp_sum = 1 - max(cp_x, cp_symmetric) + min(cp_x, cp_symmetric)
    
    # 3. Scale & sigmoid
    cp_sum *= 10                    # scaling_factor
    w = sigmoid(cp_sum - 2)         # range_th = 2
    
    # 4. Discretize
    if w > 0.9:       → return 1.2   # Strong unlearn
    elif 0.3 < w <= 0.4: → return w  # Partial unlearn
    else:              → return 0     # No unlearn (base model only)
```

#### GMM Construction (`weighting_func_gmm`)

Được fit trong `train_ood.py` lên train_scores và test_scores:

```python
gmm = GMM(n_components=2)
gmm.means_ = [[mean_train], [mean_test]]
gmm.covariances_ = [[[std_test**2]], [[std_test**2]]]  # Cả 2 dùng std của test
gmm.weights_ = [0.5, 0.5]
x0 = (mean_train + mean_test) / 2  # Decision boundary center
```

---

## 3. LLM Integration — Hacked Files

### 3.1 Tại sao cần hack?

HuggingFace Transformers + PEFT không hỗ trợ truyền **custom per-sample weight** vào LoRA layers. Cần sửa 4 file:

| File | Vai trò |
|------|---------|
| `modeling_llama_hacked_o.py` | Thêm `ood_weight` vào forward chain của LlamaForCausalLM |
| `lora_layer_hacked_o.py` | Sửa LoRA Linear layer: `result = W*x*w_base + delta_W*x*w(x)` |
| `lora_model_hacked_o.py` | Custom LoraModel dùng custom LoRA layer |
| `peft_model_hacked_o.py` | Custom PeftModel dùng custom LoraModel |

### 3.2 Forward Chain — `ood_weight` đi từ đâu đến đâu

```
LlamaForCausalLM_ood
  │  self.ood_weight (set by init_oodweight)
  │
  └─► LlamaForCausalLM_ood.forward()
        │  truyền self.ood_weight vào self.model()
        │
        └─► LlamaModel.forward()
              │  truyền ood_weight vào từng decoder_layer
              │
              └─► LlamaDecoderLayer.forward()
                    │
                    ├─► LlamaAttention.forward()
                    │     ├─► q_proj(x, ood_weight, ...)  ← LoRA Linear
                    │     ├─► k_proj(x, ood_weight, ...)
                    │     ├─► v_proj(x, ood_weight, ...)
                    │     └─► o_proj(x, ood_weight, ...)
                    │
                    └─► LlamaMLP.forward()
                          ├─► up_proj(x, ood_weight, ...)
                          ├─► gate_proj(x, ood_weight, ...)
                          └─► down_proj(x, ood_weight, ...)
```

### 3.3 LoRA Layer — Core Formula

File: `lora_layer_hacked_o.py`, function `Linear.forward()`:

```python
# Base model output
result = self.base_layer(x)                     # W * x

# LoRA output
out = lora_B(lora_A(dropout(x))) * scaling      # delta_W * x

# Soft-weighted combination
# ood_weight[0] = 1 (base weight, luôn = 1)
# ood_weight[1] = w(x) (per-sample, từ OOD detector)
w = ood_weight[1]
if isinstance(w, torch.Tensor) and w.dim() >= 1:
    w = w.view(-1, *([1] * (result.dim() - 1)))  # (B,) → (B,1,1)
result = result * ood_weight[0] + out * w
```

**Khi `w(x) = 0`**: `result = W*x` → base model thuần túy  
**Khi `w(x) = 1.2`**: `result = W*x + 1.2*delta_W*x` → LoRA unlearn kích hoạt mạnh

### 3.4 Orthogonal Loss (Training only)

Khi train LoRA mới, orthogonal loss đảm bảo LoRA mới **trực giao** với các LoRA cũ:

```python
# Trong LoRA layer forward (training):
for prev_lora_A in previous_lora_weights:
    o_a = prev_lora_A @ current_lora_A.T    # Cross-projection
    o_loss += sum(square(o_a))               # Penalize non-orthogonality
    
# Trong CausalLM forward:
total_loss = CE_loss + scale * o_loss        # scale = orthogonal_loss_weight
```

---

## 4. Inference Pipeline (eval_o3.py)

### 4.1 Setup

```python
# 1. Load LLM + LoRA
model = LlamaForCausalLM_ood.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, lora_weight)
model.init_olora(orthogonal_loss=False, olora_weights={})
model.init_active_adapters_d(active_adapters_d=['default'])

# 2. Load OOD components (per domain)
for domain in ood_types:
    ood_models.append(RobertaForSelector_inference(...))   # RoBERTa + LoRA
    ood_clrs.append(pickle.load(ocsvm))                     # OCSVM
    ood_gmm_w_cls.append(pickle.load(gmm))                  # GMM
    ood_mean_lists.append(torch.load(mean_list))             # Mean vectors
    ood_precision_lists.append(torch.load(precision_list))   # Precision matrices
    ood_fea_lists.append(torch.load(fea_list))               # Feature lists
    ood_x0.append(threshold[0])                              # GMM center x0
```

### 4.2 Per-batch Inference Loop

```python
for batch in data:
    # --- Stage 1: Tính per-sample w(x) ---
    ood_input = ood_tokenizer(batch_texts, max_length=512)
    max_ood_per_sample = zeros(batch_size)
    
    for each ood_detector:
        mah_score = ood_model.get_unsup_Mah_score_s(ood_input, ...)[:, 1:]
        ocsvm_score = ocsvm.score_samples(mah_score)         # (batch_size,)
        w = [obtain_weights(s, gmm, x0) for s in ocsvm_score]
        max_ood_per_sample = maximum(max_ood_per_sample, w)
    
    # --- Stage 2: LLM Generate với per-sample weight ---
    ood_weight_tensor = torch.tensor(max_ood_per_sample)     # (batch_size,)
    model.init_oodweight(ood_weight=[1, ood_weight_tensor])
    
    output = model.generate(input_ids=batch_input_ids)
    # Bên trong: mỗi LoRA layer broadcast w theo batch dimension
    # result = W*x * 1 + delta_W*x * w.view(B, 1, 1)
```

### 4.3 Broadcasting per-sample weight

Tại LoRA layer, khi `ood_weight[1]` là tensor `(batch_size,)`:

```python
w = ood_weight[1]                               # (B,)
w = w.view(-1, *([1] * (result.dim() - 1)))     # (B, 1, 1)
result = result * ood_weight[0] + out * w        # (B, seq, hidden) broadcast
```

Mỗi sample trong batch nhận weight riêng, không cần loop từng sample.

---

## 5. File Dependencies

```
eval_o3.py (inference entry point)
├── src/ood_model_selector.py      → RobertaForSelector_inference
├── src/peft_model_hacked_o.py     → PeftModel (custom)
│   └── src/lora_model_hacked_o.py → LoraModel (custom)
│       └── src/lora_layer_hacked_o.py → Linear.forward() (core formula)
├── src/modeling_llama_hacked_o.py  → LlamaForCausalLM_ood (ood_weight chain)
└── sklearn (OCSVM, GMM), scipy (norm.cdf)

train_ood.py (OOD training entry point)
├── src/ood_model_selector.py      → RobertaForSelector (training class)
├── src/ood_data.py                → Dataset loading
├── src/ood_utils.py               → Collate functions, detection_performance
└── src/ood_calculate_log.py       → AUROC, TNR, DTACC metrics

train_unlearn_lora_o.py (LoRA training entry point)
├── src/modeling_llama_hacked_o.py  → LlamaForCausalLM_ood
├── src/peft_model_hacked_o.py     → PeftModel
├── src/mapping_hacked_o.py        → get_peft_model (custom)
└── safetensors                     → Load previous LoRA for orthogonal loss
```

---

## 6. Hướng dẫn tái sử dụng

### Để áp dụng kiến trúc OOD này sang dự án khác, cần:

#### 6.1 Thay đổi OOD Detector
- **Giữ nguyên**: `ood_model_selector.py` (RoBERTa-based) — hoạt động với bất kỳ text input nào
- **Thay đổi**: `ood_data.py` — sửa hàm `load()` để load dataset mới
- **Thay đổi**: `train_ood.sh` — đường dẫn data và tên dataset

#### 6.2 Thay đổi LLM
- **Thay đổi**: `modeling_llama_hacked_o.py` — nếu dùng model khác Llama (ví dụ Mistral, Qwen), cần hack tương tự cho model đó: thêm `ood_weight` vào forward chain
- **Giữ nguyên**: `lora_layer_hacked_o.py` — core formula không phụ thuộc architecture
- **Giữ nguyên**: `peft_model_hacked_o.py`, `lora_model_hacked_o.py` — wrapper chung

#### 6.3 Các hyperparameter quan trọng

| Parameter | Vị trí | Default | Ý nghĩa |
|-----------|--------|---------|----------|
| `scaling_factor` | `obtain_weights()` | 10 | Scale CDF probability trước sigmoid |
| `range_th` | `obtain_weights()` | 2 | Offset sigmoid (dịch decision boundary) |
| `w > 0.9 → 1.2` | `obtain_weights()` | 0.9 / 1.2 | Threshold / multiplier cho full activation |
| `0.3 < w <= 0.4` | `obtain_weights()` | 0.3-0.4 | Partial activation range |
| OCSVM `nu` | `train_ood.py` | 0.1 | Fraction of outliers |
| `orthogonal_loss_weight` | `train_unlearn_lora_o.py` | 0.1 | Scale cho orthogonal regularization |
| `LoRA r` | `train_unlearn_lora_o.py` | 8 | LoRA rank |
| `LoRA alpha` | `train_unlearn_lora_o.py` | 16 | LoRA alpha (scaling) |

---

## 7. Tham khảo nhanh — Các hàm chính

| Hàm | File | Input → Output |
|-----|------|----------------|
| `get_unsup_Mah_score_s()` | `ood_model_selector.py` | `(batch_tokens)` → `(B, 13)` anomaly scores |
| `obtain_weights()` | `eval_o3.py` | `scalar score` → `scalar w(x)` in {0, (0.3,0.4], 1.2} |
| `weighting_func_gmm()` | `train_ood.py` | `(train_scores, test_scores)` → `(gmm, x0)` |
| `sample_X_estimator()` | `ood_model_selector.py` | `dataloader` → `(mean_list, precision_list, fea_list)` |
| `init_oodweight()` | `modeling_llama_hacked_o.py` | `[w_base, w_lora]` → set model state |
| `Linear.forward()` | `lora_layer_hacked_o.py` | `(x, ood_weight)` → `W*x*w_base + delta_W*x*w(x)` |
| `detect_ood()` | `train_ood.py` | `(model, data)` → fit OCSVM, save artifacts, return accuracy |
