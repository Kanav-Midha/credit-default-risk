"""Discrimination and calibration metrics for imbalanced binary classification.

Accuracy is meaningless here. With a 8.1% default rate, a model that predicts
"never defaults" for everyone scores 91.9% accuracy and is worth nothing. The
metrics below all measure something a lender actually cares about:

* **ROC-AUC / Gini** — ranking quality. Gini = 2*AUC - 1 is the form quoted in
  credit risk; a Gini of 0.5 is a respectable application scorecard.
* **PR-AUC** — precision/recall trade-off, which unlike ROC-AUC degrades
  visibly when the positive class is rare and hard.
* **KS statistic** — the maximum separation between the cumulative score
  distributions of defaulters and non-defaulters. The standard regulatory
  reporting metric for scorecard strength.
* **Brier score / ECE** — calibration. A model can rank perfectly and still
  output probabilities that are systematically wrong, which breaks any
  downstream expected-loss calculation.
* **Lift at k** — how much more concentrated defaults are in the riskiest k%
  of the book than in the book overall. This is what a portfolio manager asks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class ClassificationReport:
    """Container for the full metric suite, for logging and serialisation."""

    roc_auc: float
    gini: float
    pr_auc: float
    ks_statistic: float
    ks_threshold: float
    brier_score: float
    expected_calibration_error: float
    lift_at_10pct: float
    default_rate: float
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"  ROC-AUC   {self.roc_auc:.4f}   (Gini {self.gini:.4f})\n"
            f"  PR-AUC    {self.pr_auc:.4f}   (baseline {self.default_rate:.4f})\n"
            f"  KS        {self.ks_statistic:.4f}  at score {self.ks_threshold:.4f}\n"
            f"  Brier     {self.brier_score:.5f}  ECE {self.expected_calibration_error:.5f}\n"
            f"  Lift@10%  {self.lift_at_10pct:.2f}x\n"
            f"  n={self.n_samples:,}"
        )


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty array")
    if not np.isin(np.unique(y_true), [0, 1]).all():
        raise ValueError(f"y_true must be binary 0/1; got {np.unique(y_true)[:5]}")
    if np.isnan(y_prob).any():
        raise ValueError("y_prob contains NaN")
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError(f"y_prob must lie in [0, 1]; got [{y_prob.min()}, {y_prob.max()}]")
    return y_true, y_prob


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov separation between defaulter and non-defaulter scores.

    Equivalent to ``max(TPR - FPR)`` across all thresholds, which is why it can
    be read straight off the ROC curve.

    Returns:
        ``(ks, threshold)`` — the maximum separation and the score at which it
        occurs. That threshold is a useful *starting point* for a cutoff, but
        not the right one: see :mod:`credit_risk.evaluation.business`.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = int(np.argmax(tpr - fpr))
    return float(tpr[idx] - fpr[idx]), float(thresholds[idx])


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> float:
    """Mean absolute gap between predicted probability and observed frequency.

    Predictions are grouped into bins and, within each bin, the average
    predicted probability is compared with the actual default rate. The gaps
    are averaged weighted by bin size.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities of default.
        n_bins: Number of bins.
        strategy: ``"quantile"`` for equal-count bins (robust when scores are
            clustered near zero, as they are here) or ``"uniform"`` for
            equal-width bins.

    Returns:
        ECE in probability units. 0.01 means predictions are off by one
        percentage point on average.
    """
    y_true, y_prob = _validate(y_true, y_prob)

    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(y_prob.min(), y_prob.max(), n_bins + 1)
    else:
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")

    if edges.size < 2:
        return 0.0  # degenerate: all predictions identical

    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, edges.size - 2)

    total_error = 0.0
    for b in range(edges.size - 1):
        mask = bin_ids == b
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue
        total_error += n_in_bin * abs(y_prob[mask].mean() - y_true[mask].mean())

    return float(total_error / y_true.size)


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    """Concentration of defaults in the riskiest ``k`` fraction of applicants.

    A lift of 3.0 at k=0.10 means the top-decile-by-risk contains three times
    the default rate of the portfolio as a whole.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    if not 0 < k <= 1:
        raise ValueError(f"k must lie in (0, 1], got {k}")

    base_rate = y_true.mean()
    if base_rate == 0:
        return 0.0

    n_top = max(1, int(round(k * y_true.size)))
    top_idx = np.argsort(-y_prob, kind="stable")[:n_top]
    return float(y_true[top_idx].mean() / base_rate)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> ClassificationReport:
    """Compute the full metric suite in one pass."""
    y_true, y_prob = _validate(y_true, y_prob)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both classes present to evaluate discrimination")

    auc = float(roc_auc_score(y_true, y_prob))
    ks, ks_thr = ks_statistic(y_true, y_prob)

    return ClassificationReport(
        roc_auc=auc,
        gini=2 * auc - 1,
        pr_auc=float(average_precision_score(y_true, y_prob)),
        ks_statistic=ks,
        ks_threshold=ks_thr,
        brier_score=float(brier_score_loss(y_true, y_prob)),
        expected_calibration_error=expected_calibration_error(y_true, y_prob),
        lift_at_10pct=lift_at_k(y_true, y_prob, k=0.10),
        default_rate=float(y_true.mean()),
        n_samples=int(y_true.size),
    )
