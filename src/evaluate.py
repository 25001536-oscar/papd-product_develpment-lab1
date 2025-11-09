import argparse, json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from utils import load_params

def load_preprocessed(target):
    df = pd.read_parquet("artifacts/preprocessed.parquet")
    y = df[target].values
    X = df.drop(columns=[target]).values
    return X, y

def main(params_path: str):
    P = load_params(params_path)
    target = P["dataset"]["target"]

    X, y = load_preprocessed(target)
    model = joblib.load("models/best_model.joblib")

    preds = model.predict(X)
    rmse = root_mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    with open("artifacts/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # simple markdown report
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/report.md", "w", encoding="utf-8") as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"- RMSE: {rmse:.4f}\n")
        f.write(f"- MAE: {mae:.4f}\n")
        f.write(f"- R2: {r2:.4f}\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args = ap.parse_args()
    main(args.params)
