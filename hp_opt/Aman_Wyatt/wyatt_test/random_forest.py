import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ---------------------------
# Config
# ---------------------------
@dataclass
class RFConfig:
    dataset: str = "adult"  # "adult" (classification) or "california" (regression)
    test_size: float = 0.2
    val_size: float = 0.1  # fraction OF the remaining train portion
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | int | float = "sqrt"  # typical default for clf; ok for reg too
    n_jobs: int = -1
    save_path: str = "./rf_model.joblib"


# ---------------------------
# Data loading helpers
# ---------------------------

def load_adult() -> Tuple[pd.DataFrame, pd.Series]:
    # Adult Income dataset from OpenML; version=2 is the canonical cleaned split
    ds = fetch_openml("adult", version=2, as_frame=True)
    X, y = ds.data, ds.target

    # Convert target to binary 0/1 (<=50K -> 0, >50K -> 1)
    y = (y.astype(str).str.contains(">50K")).astype(int)

    # Treat '?' as missing values
    X = X.replace("?", np.nan)
    return X, y


def load_california() -> Tuple[pd.DataFrame, pd.Series]:
    ds = fetch_california_housing(as_frame=True)
    X, y = ds.data, ds.target  # y is MedHouseVal
    return X, y


# ---------------------------
# Preprocessing + model pipeline
# ---------------------------

def build_pipeline_adult(cfg: RFConfig) -> Pipeline:
    X, y = load_adult()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ], remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("rf", model),
    ])
    return X, y, pipe


def build_pipeline_california(cfg: RFConfig) -> Pipeline:
    X, y = load_california()
    num_cols = X.columns.tolist()
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols)
    ], remainder="drop")

    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("rf", model),
    ])
    return X, y, pipe


# ---------------------------
# Train / eval
# ---------------------------

def split_train_val_test(X: pd.DataFrame, y: pd.Series, cfg: RFConfig):
    # First carve out test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y if y.nunique() <= 20 else None
    )

    # Then split train/val from remaining
    val_fraction_of_trainval = cfg.val_size
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction_of_trainval, random_state=cfg.random_state,
        stratify=y_trainval if y_trainval.nunique() <= 20 else None
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_classification(cfg: RFConfig):
    X, y, pipe = build_pipeline_adult(cfg)
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, cfg)

    # Fit on training data
    pipe.fit(X_train, y_train)

    # Evaluate
    def eval_split(name, Xs, ys):
        pred = pipe.predict(Xs)
        acc = accuracy_score(ys, pred)
        f1 = f1_score(ys, pred)
        print(f"{name:>6} | accuracy={acc:.4f}  f1={f1:.4f}")
        return acc, f1

    print("\n== Adult Income (classification) ==")
    eval_split("Train", X_train, y_train)
    eval_split(" Val ", X_val, y_val)
    eval_split(" Test", X_test, y_test)

    print("\nClassification report (Validation):")
    print(classification_report(y_val, pipe.predict(X_val), digits=4))

    return pipe


def run_regression(cfg: RFConfig):
    X, y, pipe = build_pipeline_california(cfg)
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, cfg)

    # Fit on training data
    pipe.fit(X_train, y_train)

    # Evaluate
    def eval_split(name, Xs, ys):
        pred = pipe.predict(Xs)
        r2 = r2_score(ys, pred)
        mae = mean_absolute_error(ys, pred)
        print(f"{name:>6} | R2={r2:.4f}  MAE={mae:.4f}")
        return r2, mae

    print("\n== California Housing (regression) ==")
    eval_split("Train", X_train, y_train)
    eval_split(" Val ", X_val, y_val)
    eval_split(" Test", X_test, y_test)

    return pipe


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Random Forest baseline for Adult (classification) and California Housing (regression)")
    p.add_argument("--dataset", type=str, default="adult", choices=["adult", "california"], help="Which dataset to run")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.1, help="Fraction of the remaining train portion used for validation")
    p.add_argument("--random_state", type=int, default=42)

    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--max_features", type=str, default="sqrt")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--save_path", type=str, default="./rf_model.joblib")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = RFConfig(
        dataset=args.dataset,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        save_path=args.save_path,
    )

    if cfg.dataset == "adult":
        pipe = run_classification(cfg)
    else:
        pipe = run_regression(cfg)

    # Optional: persist the whole pipeline with preprocessing + model
    try:
        import joblib
        joblib.dump({"pipeline": pipe, "config": cfg.__dict__}, cfg.save_path)
        print(f"\nSaved pipeline to: {cfg.save_path}")
    except Exception as e:
        print(f"\nWarning: could not save model to {cfg.save_path}: {e}")


if __name__ == "__main__":
    main()
