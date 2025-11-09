import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from utils import load_params
from pathlib import Path

def build_preprocessor(df, scale_numeric=True, impute_strategy="median"):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if scale_numeric:
        transformer = ColumnTransformer(
            transformers=[
                ("num", 
                 make_numeric_pipeline(impute_strategy, with_scaler=True),
                 num_cols)
            ],
            remainder="drop"
        )
    else:
        transformer = ColumnTransformer(
            transformers=[
                ("num", 
                 make_numeric_pipeline(impute_strategy, with_scaler=False),
                 num_cols)
            ],
            remainder="drop"
        )
    return transformer, num_cols

def make_numeric_pipeline(impute_strategy, with_scaler: bool):
    from sklearn.pipeline import Pipeline
    steps = [("imputer", SimpleImputer(strategy=impute_strategy))]
    if with_scaler:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)

def main(params_path: str):
    P = load_params(params_path)
    data_path = P["dataset"]["path"]
    target = P["dataset"]["target"]
    scale = bool(P["preprocess"].get("scale_numeric", True))
    impute_strategy = P["preprocess"].get("impute_strategy", "median")

    df = pd.read_csv(data_path)
    assert target in df.columns, f"Target '{target}' not found in dataset."
    y = df[target]
    X = df.drop(columns=[target])

    preproc, used_cols = build_preprocessor(X, scale, impute_strategy)
    X_trans = preproc.fit_transform(X, y)

    # Persist artifacts
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save transformed arrays as parquet-like (we'll use pandas for convenience)
    X_df = pd.DataFrame(X_trans)
    X_df[target] = y.values
    X_df.to_parquet(out_dir / "preprocessed.parquet", index=False)

    # Save the preprocessor for later use
    joblib.dump({"preprocessor": preproc, "feature_names": used_cols, "target": target}, out_dir / "preprocessor.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args = ap.parse_args()
    main(args.params)
