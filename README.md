# Large Language Model Unlearning for Source Code

## 1. Set Up the Environment

```bash
conda create -n prod python=3.10
conda activate prod
pip install -r requirements.txt
```

## 2. Run PROD Training (Unlearning)

```bash
sudo nohup bash run_PROD.sh > logs/run_prod.log 2>&1 &
```

## 3. Train OOD Detector

```bash
sudo nohup bash train_ood.sh > logs/train_ood.log 2>&1 &
```

## 4. Inference with OOD-Guided Soft Weighting

```bash
sudo nohup bash eval_soft_infer.sh > logs/eval_soft_infer.log 2>&1 &
```