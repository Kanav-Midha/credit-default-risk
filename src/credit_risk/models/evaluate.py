"""Evaluate out-of-fold predictions and produce the reporting pack.

Run as ``make evaluate``, after ``make train``.

Everything here reads the OOF predictions written by training, never a model's
predictions on its own training data. Writes to ``reports/``:

* ``metrics.json`` — the full metric suite plus the chosen decision policy
* ``sensitivity.csv`` — savings across the grid of business assumptions
* ``figures/*.png`` — ROC, precision-recall, calibration, score distribution,
  cost-versus-threshold, and feature importance
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # no display in CI or over SSH

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from credit_risk.config import (
    FIGURES_DIR,
    LGD,
    MODELS_DIR,
    PROCESSED_DIR,
    PROFIT_MARGIN,
    REPORTS_DIR,
    TARGET,
)
from credit_risk.evaluation.business import (
    DecisionPolicy,
    baseline_cost,
    expected_cost,
    optimal_threshold,
    sensitivity_analysis,
)
from credit_risk.evaluation.metrics import ClassificationReport, evaluate

FIGSIZE = (7, 5)
DPI = 130


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    {path.name}")


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_idx = int(np.argmax(tpr - fpr))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(fpr, tpr, lw=2, label=f"Model (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random")
    ax.vlines(
        fpr[ks_idx],
        fpr[ks_idx],
        tpr[ks_idx],
        color="crimson",
        lw=2,
        label=f"KS = {tpr[ks_idx] - fpr[ks_idx]:.4f}",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (out-of-fold)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "roc_curve")


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray, pr_auc: float) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    base = y_true.mean()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(recall, precision, lw=2, label=f"Model (PR-AUC = {pr_auc:.4f})")
    ax.axhline(base, ls="--", color="grey", lw=1, label=f"No skill ({base:.3f})")
    ax.set_xlabel("Recall (share of defaults identified)")
    ax.set_ylabel("Precision (share of flagged that default)")
    ax.set_title("Precision-recall curve (out-of-fold)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "precision_recall_curve")


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> None:
    """Reliability diagram on quantile bins.

    Quantile bins rather than equal-width: scores are concentrated near zero,
    so equal-width bins would leave the upper half of the range almost empty
    and the curve would be dominated by noise.
    """
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1]), 0, edges.size - 2)

    predicted, observed, counts = [], [], []
    for b in range(edges.size - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        predicted.append(y_prob[mask].mean())
        observed.append(y_true[mask].mean())
        counts.append(int(mask.sum()))

    fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(7, 7), height_ratios=[3, 1], sharex=True)
    lim = max(max(predicted), max(observed)) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="grey", lw=1, label="Perfect calibration")
    ax.plot(predicted, observed, "o-", lw=2, ms=5, label="Model")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability diagram (out-of-fold)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax_hist.bar(predicted, counts, width=lim / (len(predicted) * 2), color="steelblue")
    ax_hist.set_xlabel("Predicted probability of default")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.3)
    _save(fig, "calibration")


def plot_score_distribution(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, np.quantile(y_prob, 0.999), 60)
    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.6, density=True, label="Repaid")
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.6, density=True, label="Defaulted")
    ax.axvline(threshold, color="crimson", ls="--", lw=2, label=f"Decline above {threshold:.4f}")
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution by outcome")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "score_distribution")


def plot_cost_curve(
    y_true: np.ndarray, y_prob: np.ndarray, exposure: np.ndarray, policy: DecisionPolicy
) -> None:
    """Total cost as a function of the cutoff, with 0.5 marked for contrast."""
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 150)))
    costs = [expected_cost(y_true, y_prob, exposure, t) for t in thresholds]
    base = baseline_cost(y_true, exposure)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(thresholds, np.array(costs) / 1e6, lw=2, label="Cost of policy")
    ax.axhline(base / 1e6, ls="--", color="grey", lw=1.5, label="Approve everyone")
    ax.axvline(
        policy.threshold,
        color="crimson",
        ls="--",
        lw=2,
        label=f"Optimum ({policy.threshold:.4f})",
    )
    if thresholds.min() <= 0.5 <= thresholds.max():
        ax.axvline(0.5, color="darkorange", ls=":", lw=2, label="Naive 0.5 cutoff")
    ax.set_xlabel("Decline threshold")
    ax.set_ylabel("Total expected cost (millions)")
    ax.set_title("Cost versus decision threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "cost_vs_threshold")


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 25) -> None:
    top = importance.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.3)))
    ax.barh(top["feature"], top["gain"], color="steelblue")
    ax.set_xlabel("Average gain across folds")
    ax.set_title(f"Top {top_n} features")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, "feature_importance")


def build_report(report: ClassificationReport, policy: DecisionPolicy, base: float, n: int) -> dict:
    savings = base - policy.total_cost
    return {
        "discrimination_and_calibration": report.to_dict(),
        "assumptions": {"lgd": LGD, "profit_margin": PROFIT_MARGIN},
        "decision_policy": {
            "threshold": policy.threshold,
            "approval_rate": policy.approval_rate,
            "bad_rate_of_approved": policy.bad_rate_of_approved,
            "defaults_caught": policy.defaults_caught,
            "good_customers_rejected": policy.good_customers_rejected,
        },
        "portfolio_impact": {
            "cost_approve_everyone": base,
            "cost_under_policy": policy.total_cost,
            "savings": savings,
            "savings_pct": savings / base if base else 0.0,
            "savings_per_applicant": savings / n,
        },
    }


def main() -> None:
    oof_path = MODELS_DIR / "oof_predictions.parquet"
    if not oof_path.is_file():
        raise FileNotFoundError(f"{oof_path} not found. Run `make train` first.")

    oof = pd.read_parquet(oof_path)
    y_true = oof[TARGET].to_numpy()
    y_prob = oof["prediction"].to_numpy()

    # Exposure at risk drives the cost model, so it comes from the actual
    # credit amount rather than a flat per-applicant assumption.
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet", columns=["SK_ID_CURR", "AMT_CREDIT"])
    exposure = (
        oof[["SK_ID_CURR"]]
        .merge(train, on="SK_ID_CURR", how="left")["AMT_CREDIT"]
        .fillna(train["AMT_CREDIT"].median())
        .to_numpy()
    )

    report = evaluate(y_true, y_prob)
    print(f"\nDiscrimination and calibration:\n{report}\n")

    policy = optimal_threshold(y_true, y_prob, exposure)
    base = baseline_cost(y_true, exposure)
    print(f"Cost-optimal decision policy:\n{policy}")
    print(f"  savings vs approve-all {(base - policy.total_cost) / base:.2%}\n")

    print("  figures:")
    plot_roc(y_true, y_prob, report.roc_auc)
    plot_precision_recall(y_true, y_prob, report.pr_auc)
    plot_calibration(y_true, y_prob)
    plot_score_distribution(y_true, y_prob, policy.threshold)
    plot_cost_curve(y_true, y_prob, exposure, policy)

    importance_path = MODELS_DIR / "feature_importance.csv"
    if importance_path.is_file():
        plot_feature_importance(pd.read_csv(importance_path))

    metrics = build_report(report, policy, base, len(y_true))
    with (REPORTS_DIR / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    sensitivity = pd.DataFrame(sensitivity_analysis(y_true, y_prob, exposure))
    sensitivity.to_csv(REPORTS_DIR / "sensitivity.csv", index=False)

    print("\n  reports/metrics.json")
    print("  reports/sensitivity.csv")
    print(
        f"\n  Savings hold across {(sensitivity['savings'] > 0).sum()}/"
        f"{len(sensitivity)} assumption combinations."
    )


if __name__ == "__main__":
    main()
