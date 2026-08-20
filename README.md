# Large Language Model Unlearning for Source Code

This repository contains the code and data for the paper titled **Large Language Model Unlearning for Source Code**.

### 1. Set Up the Environment

Install **Conda 3.10** and create the environment:

```bash
conda create -n prod python=3.10
conda activate prod
```

Install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install python-dateutil
pip install evaluate
```

### 2. Run the Experiments

Execute the `run_PROD.sh` script to run the experiments:

```bash
bash run_PROD.sh
```