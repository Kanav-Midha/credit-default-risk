"""Translating predicted probabilities into a lending decision.

A classifier outputs a probability. A lender needs a yes/no. Converting one to
the other at the default 0.5 cutoff is the single most common mistake in
student credit-risk projects: at an 8% base rate, almost nothing scores above
0.5, so the model approves everyone and adds no value.

The correct cutoff comes from the *asymmetry of the two errors*:

* **Accepting a bad applicant** costs ``LGD * exposure`` — the principal that
  is not recovered after default.
* **Rejecting a good applicant** costs ``margin * exposure`` — the profit that
  would have been earned on a loan that would have performed.

Because LGD (~0.65) is several times the margin (~0.12), false negatives are
far more expensive than false positives, and the optimal threshold sits well
below 0.5. This module finds it by direct search over the validation set.

The LGD and margin values are *assumptions*, documented in
``docs/methodology.md``. Results are reported alongside a sensitivity analysis
so the conclusion does not rest on a single guessed constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from credit_risk.config import LGD, PROFIT_MARGIN


@dataclass(frozen=True)
class DecisionPolicy:
    """A threshold plus the portfolio outcome it produces."""

    threshold: float
    total_cost: float
    cost_per_applicant: float
    approval_rate: float
    bad_rate_of_approved: float
    defaults_caught: float
    good_customers_rejected: float

    def __str__(self) -> str:
        return (
            f"  threshold          {self.threshold:.4f}\n"
            f"  approval rate      {self.approval_rate:.1%}\n"
            f"  bad rate (approved){self.bad_rate_of_approved:.2%}\n"
            f"  defaults caught    {self.defaults_caught:.1%}\n"
            f"  good cust. lost    {self.good_customers_rejected:.1%}\n"
            f"  cost per applicant {self.cost_per_applicant:,.0f}"
        )


def expected_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    exposure: np.ndarray,
    threshold: float,
    lgd: float = LGD,
    margin: float = PROFIT_MARGIN,
) -> float:
    """Total expected cost of applying ``threshold`` to this population.

    Applicants scoring at or above the threshold are declined.

    Args:
        y_true: 1 if the applicant defaulted, 0 otherwise.
        y_prob: Predicted probability of default.
        exposure: Credit amount at risk per applicant, in currency units.
        threshold: Decline applicants with ``y_prob >= threshold``.
        lgd: Loss given default, as a fraction of exposure.
        margin: Profit on a performing loan, as a fraction of exposure.

    Returns:
        Total cost in currency units: unrecovered principal on approved
        defaulters, plus forgone profit on declined good applicants.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    exposure = np.asarray(exposure, dtype=float).ravel()

    if not (y_true.shape == y_prob.shape == exposure.shape):
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_prob {y_prob.shape}, "
            f"exposure {exposure.shape}"
        )

    approved = y_prob < threshold

    loss_from_approved_bads = float((exposure * lgd)[approved & (y_true == 1)].sum())
    forgone_profit_on_rejected_goods = float((exposure * margin)[~approved & (y_true == 0)].sum())
    return loss_from_approved_bads + forgone_profit_on_rejected_goods


def _policy_at(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    exposure: np.ndarray,
    threshold: float,
    lgd: float,
    margin: float,
) -> DecisionPolicy:
    approved = y_prob < threshold
    n = y_true.size
    n_approved = int(approved.sum())
    n_bad = int((y_true == 1).sum())
    n_good = int((y_true == 0).sum())

    cost = expected_cost(y_true, y_prob, exposure, threshold, lgd, margin)

    return DecisionPolicy(
        threshold=float(threshold),
        total_cost=cost,
        cost_per_applicant=cost / n,
        approval_rate=n_approved / n,
        bad_rate_of_approved=(float(y_true[approved].mean()) if n_approved else 0.0),
        defaults_caught=(float(((~approved) & (y_true == 1)).sum() / n_bad) if n_bad else 0.0),
        good_customers_rejected=(
            float(((~approved) & (y_true == 0)).sum() / n_good) if n_good else 0.0
        ),
    )


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    exposure: np.ndarray,
    lgd: float = LGD,
    margin: float = PROFIT_MARGIN,
    n_candidates: int = 200,
) -> DecisionPolicy:
    """Find the cutoff that minimises total expected cost.

    Candidate thresholds are taken from quantiles of the predicted scores
    rather than an even grid over [0, 1]: scores are heavily concentrated near
    zero, so an even grid would waste almost all of its evaluations in a region
    containing no applicants.

    Args:
        n_candidates: Number of quantile-spaced thresholds to evaluate.

    Returns:
        The best :class:`DecisionPolicy` found.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    exposure = np.asarray(exposure, dtype=float).ravel()

    candidates = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, n_candidates)))
    policies = [_policy_at(y_true, y_prob, exposure, t, lgd, margin) for t in candidates]
    return min(policies, key=lambda p: p.total_cost)


def baseline_cost(
    y_true: np.ndarray,
    exposure: np.ndarray,
    lgd: float = LGD,
) -> float:
    """Cost of the do-nothing policy: approve every applicant.

    This is the benchmark the model has to beat. Quoting model savings against
    anything else overstates the result.
    """
    y_true = np.asarray(y_true).ravel()
    exposure = np.asarray(exposure, dtype=float).ravel()
    return float((exposure * lgd)[y_true == 1].sum())


def sensitivity_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    exposure: np.ndarray,
    lgd_grid: tuple[float, ...] = (0.45, 0.55, 0.65, 0.75, 0.85),
    margin_grid: tuple[float, ...] = (0.08, 0.10, 0.12, 0.15, 0.20),
) -> list[dict[str, float]]:
    """Re-optimise the threshold across a grid of business assumptions.

    The point is to show *how much the conclusion depends on the guessed
    constants*. If savings stay positive across the whole grid, the result is
    robust; if it flips sign, the headline number is an artefact of the
    assumption rather than a property of the model.

    Returns:
        One record per (lgd, margin) pair with the optimal threshold, the
        savings versus approving everyone, and the resulting approval rate.
    """
    records: list[dict[str, float]] = []
    for lgd in lgd_grid:
        for margin in margin_grid:
            policy = optimal_threshold(y_true, y_prob, exposure, lgd=lgd, margin=margin)
            base = baseline_cost(y_true, exposure, lgd=lgd)
            records.append(
                {
                    "lgd": lgd,
                    "margin": margin,
                    "threshold": policy.threshold,
                    "approval_rate": policy.approval_rate,
                    "savings": base - policy.total_cost,
                    "savings_pct": (base - policy.total_cost) / base if base else 0.0,
                }
            )
    return records
