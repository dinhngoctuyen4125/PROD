### 1. Clone the Repository

```bash
git clone https://github.com/dinhngoctuyen4125/PROD.git
cd PROD
```

### 2. Set Up the Python Environment

```bash
conda create -n prod_env python=3.13.11 -y
conda activate prod_env
python -m pip install --upgrade pip
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Experiments

```bash
bash run_PROD.sh
```