import argparse, json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import linear_model, ensemble
from utils import load_params

def load_preprocessed():
    df = pd.read_parquet("artifacts/preprocessed.parquet")
    target = joblib.load("artifacts/preprocessor.joblib")["target"]
    y = df[target].values
    X = df.drop(columns=[target]).values
    return X, y

def get_estimator(estimator_path: str):
    # Import estimator by name from sklearn.* path
    module_name, class_name = estimator_path.rsplit(".", 1)
    module = __import__("sklearn." + module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls()

def main(params_path: str):
    P = load_params(params_path)
    models_cfg = P["models"]
    scoring = P["training"].get("scoring", "neg_root_mean_squared_error")
    cv_folds = int(P["training"].get("cv_folds", 5))

    X, y = load_preprocessed()

    results = []
    best = {"score": -np.inf, "name": None, "best_params": None}

    for m in models_cfg:
        name = m["name"]
        est_path = m["estimator"]
        param_grid = m.get("params", {})
        # Prefix grid keys with 'est__' since we wrap estimator in pipeline later if needed
        # Here data already preprocessed, so we can pass estimator directly
        est = get_estimator(est_path)

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        gs = GridSearchCV(est, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
        gs.fit(X, y)

        best_score = gs.best_score_
        results.append({"name": name, "best_score": float(best_score), "best_params": gs.best_params_})

        if best_score > best["score"]:
            best = {"score": float(best_score), "name": name, "best_params": gs.best_params_, "estimator_path": est_path, "estimator": gs.best_estimator_}

    # Persist the best model
    Path("models").mkdir(exist_ok=True, parents=True)
    joblib.dump(best["estimator"], "models/best_model.joblib")

    # Save train summary for evaluate stage
    Path("artifacts").mkdir(exist_ok=True, parents=True)
    with open("artifacts/train_summary.json", "w", encoding="utf-8") as f:
        json.dump({"scoring": scoring, "cv_folds": cv_folds, "results": results, "best": {"name": best["name"], "score": best["score"], "best_params": best["best_params"]}}, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args = ap.parse_args()
    main(args.params)
