# Large Language Model Unlearning for Source Code

## 1. Set Up the Environment

```bash
conda create -n prod python=3.10
conda activate prod
pip install -r requirements.txt
```

## 2. Run PROD Training (Unlearning)

```bash
bash run_PROD.sh
```

## 3. Train OOD Detector

```bash
bash train_ood.sh
```

## 4. Inference with OOD-Guided Soft Weighting

```bash
bash eval_soft_infer.sh
```