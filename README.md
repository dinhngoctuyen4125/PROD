# Large Language Model Unlearning for Source Code

This repository contains the code and data for the paper titled **Large Language Model Unlearning for Source Code**.

### 1. Set Up the Environment

Install **Conda 3.12.13** and create the environment:

```bash
conda create -n prod python=3.12.13
conda activate prod
```

Install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

### 2. Run the Experiments

Execute the `run_PROD.sh` script to run the experiments:

```bash
bash run_PROD.sh
```