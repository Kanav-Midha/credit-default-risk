"""Cross-validated LightGBM training.

Run as ``make train``.

The training loop produces **out-of-fold predictions**: for every row in the
training set, a prediction made by a model that never saw that row. Evaluating
on those is the only honest single-number estimate of performance available
without burning a holdout set, and it is what every metric in this project is
computed on.

Artefacts written to ``models/``:

* ``lgbm_fold{k}.txt`` — one booster per fold, for inference-time averaging
* ``oof_predictions.parquet`` — the OOF scores, for threshold and calibration work
* ``feature_importance.csv`` — gain, averaged across folds
* ``cv_metrics.json`` — per-fold and aggregate metrics
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from credit_risk.config import (
    CONFIG_DIR,
    ID_COL,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    TARGET,
)
from credit_risk.evaluation.metrics import evaluate

#: Columns that must never be fed to the model. ``SK_ID_CURR`` is an
#: administrative key; if it correlates with the target at all, that is an
#: artefact of how the data was assembled, not a real effect, and a tree will
#: happily memorise it.
LEAKY_COLUMNS = {ID_COL, TARGET, "index"}


@dataclass
class CVResult:
    """Everything produced by a cross-validation run."""

    oof_predictions: np.ndarray
    fold_assignments: np.ndarray
    models: list[lgb.Booster]
    feature_importance: pd.DataFrame
    fold_scores: list[float]
    best_iterations: list[int]
    feature_names: list[str]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load ``configs/model.yaml``."""
    path = path or CONFIG_DIR / "model.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def select_features(df: pd.DataFrame) -> list[str]:
    """Return the modelling columns, excluding keys and the target.

    Raises:
        ValueError: If no usable feature columns remain.
    """
    features = [c for c in df.columns if c not in LEAKY_COLUMNS]
    if not features:
        raise ValueError("No feature columns remain after exclusions")
    return features


def train_cv(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    verbose: bool = True,
) -> CVResult:
    """Train one LightGBM per stratified fold and collect OOF predictions.

    Folds are **stratified** on the target: with an 8% positive rate, an
    unstratified 5-way split can easily hand one fold a materially different
    default rate, making fold scores incomparable and the mean unstable.

    Args:
        df: Processed training table, including ``TARGET``.
        config: Parsed ``model.yaml``. Loaded from disk when omitted.
        verbose: Print per-fold progress.

    Returns:
        A :class:`CVResult`.

    Raises:
        KeyError: If ``TARGET`` is absent from ``df``.
    """
    if TARGET not in df.columns:
        raise KeyError(f"{TARGET!r} not found — was this the test split?")

    config = config or load_config()
    params = dict(config["lightgbm"])
    train_cfg = config["training"]

    n_folds = int(train_cfg["n_folds"])
    n_estimators = int(params.pop("n_estimators"))
    params["seed"] = RANDOM_SEED

    features = select_features(df)
    X = df[features]
    y = df[TARGET].to_numpy()

    categorical = [c for c in features if isinstance(X[c].dtype, pd.CategoricalDtype)]

    if verbose:
        print(
            f"Training on {len(X):,} rows x {len(features)} features "
            f"({len(categorical)} categorical)\n"
            f"Default rate {y.mean():.4%} | {n_folds}-fold stratified CV\n"
        )

    oof = np.zeros(len(df), dtype=float)
    folds = np.full(len(df), -1, dtype=int)
    models: list[lgb.Booster] = []
    fold_scores: list[float] = []
    best_iters: list[int] = []
    importances = np.zeros(len(features), dtype=float)

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    t0 = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        train_set = lgb.Dataset(X.iloc[train_idx], y[train_idx], categorical_feature=categorical)
        valid_set = lgb.Dataset(X.iloc[valid_idx], y[valid_idx], categorical_feature=categorical)

        booster = lgb.train(
            params,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(int(train_cfg["early_stopping_rounds"]), verbose=False),
                lgb.log_evaluation(int(train_cfg["log_evaluation_period"]) if verbose else 0),
            ],
        )

        oof[valid_idx] = booster.predict(X.iloc[valid_idx], num_iteration=booster.best_iteration)
        folds[valid_idx] = fold
        models.append(booster)
        best_iters.append(booster.best_iteration)
        importances += booster.feature_importance(importance_type="gain")

        from sklearn.metrics import roc_auc_score

        score = float(roc_auc_score(y[valid_idx], oof[valid_idx]))
        fold_scores.append(score)

        if verbose:
            print(f"  fold {fold}/{n_folds}  AUC {score:.5f}  best_iter {booster.best_iteration:,}")

    importance_df = (
        pd.DataFrame({"feature": features, "gain": importances / n_folds})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        elapsed = time.perf_counter() - t0
        print(
            f"\n  CV AUC {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}"
            f"   ({elapsed:.0f}s)"
        )

    return CVResult(
        oof_predictions=oof,
        fold_assignments=folds,
        models=models,
        feature_importance=importance_df,
        fold_scores=fold_scores,
        best_iterations=best_iters,
        feature_names=features,
    )


def save_artefacts(result: CVResult, df: pd.DataFrame, output_dir: Path = MODELS_DIR) -> None:
    """Persist boosters, OOF predictions, importances and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, booster in enumerate(result.models, start=1):
        booster.save_model(
            str(output_dir / f"lgbm_fold{i}.txt"), num_iteration=booster.best_iteration
        )

    oof_frame = pd.DataFrame(
        {
            ID_COL: df[ID_COL].to_numpy(),
            TARGET: df[TARGET].to_numpy(),
            "prediction": result.oof_predictions,
            "fold": result.fold_assignments,
        }
    )
    oof_frame.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    result.feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    report = evaluate(df[TARGET].to_numpy(), result.oof_predictions)
    metrics = {
        "cv_auc_mean": float(np.mean(result.fold_scores)),
        "cv_auc_std": float(np.std(result.fold_scores)),
        "fold_scores": result.fold_scores,
        "best_iterations": result.best_iterations,
        "n_features": len(result.feature_names),
        "oof": report.to_dict(),
    }
    with (output_dir / "cv_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOut-of-fold performance:\n{report}")
    print(f"\nArtefacts written to {output_dir}")


def main() -> None:
    """Train on the processed training table and save everything."""
    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.is_file():
        raise FileNotFoundError(f"{train_path} not found. Run `make features` first.")

    df = pd.read_parquet(train_path)
    result = train_cv(df)
    save_artefacts(result, df)

    print("\nTop 20 features by gain:")
    print(result.feature_importance.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
