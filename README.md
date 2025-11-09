# AutoML with DVC — Lab Scaffold

This repository contains a minimal, reproducible AutoML-ish pipeline managed with **DVC** and **Git**. It follows a 3-stage pipeline:

1. **preprocess** — cleans and scales numeric features and stores a preprocessed parquet.
2. **train** — runs a simple grid search across models and saves the best model.
3. **evaluate** — evaluates the best model, writes `artifacts/metrics.json`, and a short `reports/report.md`.

> Dataset used by default: `data/dataset_v1.csv` (California Housing), target `MedHouseVal`.

## Quick start

```bash
# 1) clone your repo or create a new one
git init
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) init DVC
dvc init

# 3) add data with DVC (large files should NOT be committed to git)
dvc add data/dataset_v1.csv
git add data/dataset_v1.csv.dvc .gitignore dvc.yaml params.yaml src requirements.txt
git commit -m "Init: DVC pipeline + dataset v1"

# 4) run the pipeline
dvc repro

# 5) show metrics
dvc metrics show
```

## Evolving datasets

To create `dataset_v2.csv` (e.g., after cleaning or transformations), place it under `data/`, run:

```bash
dvc add data/dataset_v2.csv
git add data/dataset_v2.csv.dvc
git commit -m "Dataset v2: cleaned data"
```

Then edit `params.yaml` to point to `data/dataset_v2.csv` under `dataset.path`, and re-run:

```bash
dvc repro
dvc metrics diff  # compare metrics between Git revisions
```

## Configuration

See `params.yaml` — you can set:

- dataset path and target
- preprocessing options (imputation, scaling)
- list of models and their grids
- CV folds and scoring

## Outputs

- `artifacts/preprocessed.parquet`
- `models/best_model.joblib`
- `artifacts/metrics.json`
- `reports/report.md`
```

