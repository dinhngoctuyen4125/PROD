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