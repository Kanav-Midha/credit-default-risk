"""Features from the credit bureau tables.

``bureau.csv`` records every credit the applicant holds at *other* institutions,
as reported to the national credit bureau. ``bureau_balance.csv`` adds a monthly
repayment-status snapshot for each of those credits.

This is the richest external signal available at application time: it shows how
the applicant handles debt nobody at Home Credit can see. The chain is

    bureau_balance  --(SK_ID_BUREAU)-->  bureau  --(SK_ID_CURR)-->  application

so the collapse happens twice: monthly rows down to one row per credit, then
credits down to one row per applicant.

**Why no target statistics here.** A tempting feature is "average default rate
of applicants with a similar bureau profile". Computed naively over the whole
training set, that leaks the target into every row and inflates validation
scores by several points while adding nothing on unseen data. Every feature in
this module is a function of the applicant's own history only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import BUREAU_ID, ID_COL
from credit_risk.features.aggregation import aggregate_by_key, one_hot_counts

#: ``bureau_balance.STATUS`` codes. 'C' = closed, 'X' = unknown, and '0'-'5'
#: are days-past-due buckets, where '5' means written off.
DPD_STATUSES = ["1", "2", "3", "4", "5"]

_BALANCE_AGGS: dict[str, list[str]] = {
    "MONTHS_BALANCE": ["min", "max", "size"],
}

_BUREAU_AGGS: dict[str, list[str]] = {
    # Recency and spread of credit-seeking behaviour.
    "DAYS_CREDIT": ["min", "max", "mean", "var"],
    "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
    "DAYS_CREDIT_UPDATE": ["mean"],
    # Delinquency.
    "CREDIT_DAY_OVERDUE": ["max", "mean"],
    "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
    "AMT_CREDIT_SUM_OVERDUE": ["max", "mean", "sum"],
    # Exposure.
    "AMT_CREDIT_SUM": ["max", "mean", "sum"],
    "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
    "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
    "AMT_ANNUITY": ["max", "mean", "sum"],
    # Renegotiation is a distress signal.
    "CNT_CREDIT_PROLONG": ["max", "sum"],
}


def aggregate_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Collapse monthly status rows to one row per bureau credit.

    Args:
        bureau_balance: Raw ``bureau_balance`` table.

    Returns:
        One row per ``SK_ID_BUREAU``, indexed by it.
    """
    status_counts = one_hot_counts(
        bureau_balance, key=BUREAU_ID, column="STATUS", prefix="BB", normalise=True
    )
    numeric = aggregate_by_key(
        bureau_balance, key=BUREAU_ID, aggregations=_BALANCE_AGGS, prefix="BB"
    )

    out = numeric.join(status_counts, how="left")
    out = out.rename(columns={"BB_MONTHS_BALANCE_SIZE": "BB_MONTHS_COUNT"})

    # Any month in a days-past-due bucket is a delinquency event. Collapsing
    # the individual buckets into one count gives the model a low-variance
    # summary alongside the sparse per-bucket columns.
    dpd_cols = [f"BB_STATUS_{s}_COUNT" for s in DPD_STATUSES if f"BB_STATUS_{s}_COUNT" in out]
    if dpd_cols:
        out["BB_DPD_MONTHS"] = out[dpd_cols].sum(axis=1)
        months = out["BB_MONTHS_COUNT"].replace(0, np.nan)
        out["BB_DPD_RATIO"] = out["BB_DPD_MONTHS"] / months

    return out


def build_bureau_features(
    bureau: pd.DataFrame, bureau_balance: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build one row of bureau features per applicant.

    Args:
        bureau: Raw ``bureau`` table.
        bureau_balance: Raw ``bureau_balance`` table. Optional — the bureau
            aggregates are still useful without the monthly history, which
            makes it easy to test this function on a small fixture.

    Returns:
        One row per ``SK_ID_CURR``, indexed by it.
    """
    df = bureau.copy()

    if bureau_balance is not None:
        balance_features = aggregate_bureau_balance(bureau_balance)
        df = df.merge(balance_features, how="left", left_on=BUREAU_ID, right_index=True)

    # Per-credit ratio, computed before collapsing so that each credit
    # contributes its own utilisation rather than a portfolio-level average.
    df["BUREAU_DEBT_CREDIT_RATIO"] = df["AMT_CREDIT_SUM_DEBT"] / df["AMT_CREDIT_SUM"].replace(
        0, np.nan
    )
    df["BUREAU_DEBT_CREDIT_RATIO"] = df["BUREAU_DEBT_CREDIT_RATIO"].replace(
        [np.inf, -np.inf], np.nan
    )

    aggs = dict(_BUREAU_AGGS)
    aggs["BUREAU_DEBT_CREDIT_RATIO"] = ["mean", "max"]
    if bureau_balance is not None:
        for col in ("BB_MONTHS_COUNT", "BB_DPD_MONTHS", "BB_DPD_RATIO"):
            if col in df.columns:
                aggs[col] = ["mean", "max", "sum"]

    present_aggs = {col: fns for col, fns in aggs.items() if col in df.columns}
    out = aggregate_by_key(df, key=ID_COL, aggregations=present_aggs, prefix="BUREAU")

    out["BUREAU_CREDIT_COUNT"] = df.groupby(ID_COL).size()

    # Categorical composition of the applicant's bureau file.
    for col, prefix in (("CREDIT_ACTIVE", "BUREAU"), ("CREDIT_TYPE", "BUREAU")):
        if col in df.columns:
            out = out.join(one_hot_counts(df, ID_COL, col, prefix), how="left")

    # --- Portfolio-level ratios, computed after the collapse ----------------
    total_credit = out.get("BUREAU_AMT_CREDIT_SUM_SUM")
    total_debt = out.get("BUREAU_AMT_CREDIT_SUM_DEBT_SUM")
    if total_credit is not None and total_debt is not None:
        out["BUREAU_TOTAL_DEBT_CREDIT_RATIO"] = (
            total_debt / total_credit.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)

    # Credit-seeking intensity: credits opened per year of bureau history.
    # A burst of recent applications is a classic pre-default pattern.
    if "BUREAU_DAYS_CREDIT_MIN" in out.columns:
        history_years = (-out["BUREAU_DAYS_CREDIT_MIN"] / 365.25).replace(0, np.nan)
        out["BUREAU_CREDITS_PER_YEAR"] = out["BUREAU_CREDIT_COUNT"] / history_years

    return out
