import argparse
import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from random_forest import RFConfig, run_classification, run_regression

from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

def _build_cfg(rf_mod, base_kwargs: Dict[str,Any]):
    RFConfig = getattr(rf_mod,"RFConfig")
    return RFConfig(**base_kwargs)

def _train_eval_once(rf_mod, dataset: str, hp: Dict[str, Any], random_state: int, n_jobs: int) -> Dict[str, float]:
    cfg = _build_cfg(
        rf_mod,
        {
            "dataset": dataset,
            "random_state": random_state,
            "n_jobs": n_jobs,
            # hyperparameters under search
            "n_estimators": int(hp["n_estimators"]),
            "max_depth": (None if hp["max_depth"] is None else int(hp["max_depth"])),
            "min_samples_split": int(hp["min_samples_split"]),
            "min_samples_leaf": int(hp["min_samples_leaf"]),
            "max_features": hp["max_features"],
        },
    )

    if dataset == "adult":
        X,y,pipe = rf_mod.build_pipeline_adult(cfg)
    else:
        X,y,pipe = rf_mod.build_pipeline_california(cfg)

    X_train, y_train, X_val, y_val, X_test, y_test = rf_mod.split_train_val_test(X,y,cfg)
    pipe.fit(X_train,y_train)

    # evaluate on validation
    y_val_pred = pipe.predict(X_val)
    if dataset == "adult":
        metrics = {
            "f1": float(f1_score(y_val, y_val_pred)),
            "accuracy": float(accuracy_score(y_val, y_val_pred))
        }
    else: 
        metrics = {
            "r2": float(r2_score(y_val,y_val_pred)),
            "mae": float(mean_absolute_error(y_val,y_val_pred))
        }

    return metrics

def make_trainable(rf_mod, dataset: str, random_state: int, n_jobs: int):
    def _trainable(config: Dict[str,Any]):
        metrics = _train_eval_once(
            rf_mod=rf_mod,
            dataset=dataset,
            hp=config,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        if not metrics:
            raise RuntimerError("No metrics returned")
        tune.report(metrics)
    return _trainable

def main():
    parser = argparse.ArgumentParser(description="Ray Tune + Optuna for RF baselines")
    parser.add_argument("--rf-module", type=str, required=True, help="Module name where RFConfig/build_pipline_* are defined (e.g., baseline_rf)")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult","california"], help="Which dataset to optimize")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of HPO trials")
    parser.add_argument("--cpus-per-trial", type=int, default=4, help="CPU resources per trial")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    rf_mod = importlib.import_module(args.rf_module)

    param_space = {
        "n_estimators": tune.randint(100,600),
        "max_depth": tune.choice([None, 5, 10, 15, 20, 30, 40]),
        "min_samples_split": tune.randint(2, 16), 
        "min_samples_leaf": tune.randint(1,8),
        "max_features": tune.choice(["sqrt", "log2", None, 0.3, 0.5, 0.7, 1.0]),
    }

    metric = "f1" if args.dataset == "adult" else "r2"

    search_alg = OptunaSearch(metric=metric, mode="max")
    scheduler = ASHAScheduler(metric=metric, mode="max", max_t=1, grace_period=1, reduction_factor=2)
    trainable = make_trainable(rf_mod, dataset=args.dataset, random_state=args.random_state, n_jobs=args.cpus_per_trial)
    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": args.cpus_per_trial, "gpu": 1}),
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=args.num_samples,
        ),
        param_space=param_space
    )

    results = tuner.fit()

    if not results:
        print("no trilas run")
        return

    best = results.get_best_result(metric=metric, mode="max")


    print("\nBest Trial:")
    print("Config:", best.config)
    print(f"{metric}:", best.metrics[metric])

if __name__ == "__main__":
    main()