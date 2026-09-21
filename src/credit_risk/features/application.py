"""Features from the main application table.

Two jobs here, in order:

1. **Fix the data.** The raw table contains sentinel values that are not
   missing-value markers pandas recognises. Feeding them to a model as real
   numbers is silently destructive — ``DAYS_EMPLOYED`` has 55k rows encoded as
   365243, i.e. a thousand years of employment. Left alone, that single column
   poisons every split that uses it.

2. **Build ratios.** Gradient boosting splits on one feature at a time, so it
   cannot express ``AMT_CREDIT / AMT_INCOME_TOTAL`` no matter how deep it
   grows. Ratios that a credit analyst would compute by hand are exactly the
   features that trees cannot discover for themselves, and they are
   consistently the highest-value additions to this dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: ``DAYS_EMPLOYED`` uses this value to mean "not employed" (pensioners,
#: unemployed). It is 1000 years in days and must never be treated as numeric.
DAYS_EMPLOYED_SENTINEL = 365_243

#: Columns where a sentinel value stands in for missingness.
_SENTINELS: dict[str, float] = {
    "DAYS_EMPLOYED": DAYS_EMPLOYED_SENTINEL,
}

EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, mapping division-by-zero to NaN rather than +/-inf.

    LightGBM tolerates NaN natively but treats inf as an extreme finite value,
    which distorts split points and can dominate a tree.
    """
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel encodings with NaN and drop unusable rows.

    Each sentinel gets a companion boolean flag before being nulled: the fact
    that a value was sentinel-encoded is itself predictive (an unemployed
    applicant is a different risk from one whose tenure is merely unknown), and
    that signal would be lost if we only wrote NaN.

    Args:
        df: Raw ``application_train`` or ``application_test``.

    Returns:
        A cleaned copy.
    """
    out = df.copy()

    for col, sentinel in _SENTINELS.items():
        if col in out.columns:
            mask = out[col] == sentinel
            out[f"{col}_IS_SENTINEL"] = mask.astype(np.int8)
            out.loc[mask, col] = np.nan

    # 'XNA' gender: 4 rows out of 307,511. Too few to learn a category from,
    # and it is a data-entry artefact rather than a real category.
    if "CODE_GENDER" in out.columns:
        out = out[out["CODE_GENDER"] != "XNA"].copy()

    # Non-positive income or credit is impossible and would make every ratio
    # built from them meaningless.
    for col in ("AMT_INCOME_TOTAL", "AMT_CREDIT"):
        if col in out.columns:
            out.loc[out[col] <= 0, col] = np.nan

    return out


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add affordability, tenure and external-score features.

    Every feature here answers a question a human underwriter would ask.

    Returns:
        A copy with the engineered columns appended.
    """
    out = df.copy()

    # --- Affordability: can this applicant service the debt? -----------------
    out["CREDIT_INCOME_RATIO"] = _safe_ratio(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["ANNUITY_INCOME_RATIO"] = _safe_ratio(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["CREDIT_TERM"] = _safe_ratio(out["AMT_ANNUITY"], out["AMT_CREDIT"])
    out["GOODS_CREDIT_RATIO"] = _safe_ratio(out["AMT_GOODS_PRICE"], out["AMT_CREDIT"])

    # Borrowing more than the goods are worth signals that part of the loan is
    # effectively unsecured cash.
    out["CREDIT_GOODS_DIFF"] = out["AMT_CREDIT"] - out["AMT_GOODS_PRICE"]

    # --- Household: income has to stretch across dependants ------------------
    if "CNT_FAM_MEMBERS" in out.columns:
        out["INCOME_PER_PERSON"] = _safe_ratio(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"])
    if "CNT_CHILDREN" in out.columns and "CNT_FAM_MEMBERS" in out.columns:
        out["CHILDREN_RATIO"] = _safe_ratio(out["CNT_CHILDREN"], out["CNT_FAM_MEMBERS"])

    # --- Tenure and life stage ----------------------------------------------
    # DAYS_* are negative offsets from the application date; negate for
    # readability so that larger means older / longer.
    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25
    out["EMPLOYED_YEARS"] = -out["DAYS_EMPLOYED"] / 365.25
    out["EMPLOYED_AGE_RATIO"] = _safe_ratio(out["DAYS_EMPLOYED"], out["DAYS_BIRTH"])

    if "DAYS_REGISTRATION" in out.columns:
        out["REGISTRATION_AGE_RATIO"] = _safe_ratio(out["DAYS_REGISTRATION"], out["DAYS_BIRTH"])
    if "DAYS_ID_PUBLISH" in out.columns:
        out["ID_PUBLISH_AGE_RATIO"] = _safe_ratio(out["DAYS_ID_PUBLISH"], out["DAYS_BIRTH"])

    # --- External credit bureau scores --------------------------------------
    # These three normalised scores from outside agencies are the strongest
    # single predictors in the dataset. Each is missing for a different slice
    # of applicants, so summarising across them recovers signal that any one
    # column alone would lose to NaN.
    present = [c for c in EXT_SOURCE_COLS if c in out.columns]
    if present:
        ext = out[present]
        out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
        out["EXT_SOURCE_STD"] = ext.std(axis=1)
        out["EXT_SOURCE_MIN"] = ext.min(axis=1)
        out["EXT_SOURCE_MAX"] = ext.max(axis=1)
        out["EXT_SOURCE_NA_COUNT"] = ext.isna().sum(axis=1).astype(np.int8)

        # Interaction with age: a weak external score means more for a young
        # applicant with a short credit file than for an older one.
        out["EXT_SOURCE_MEAN_x_AGE"] = out["EXT_SOURCE_MEAN"] * out["AGE_YEARS"]

    # --- Document and contact completeness ----------------------------------
    doc_cols = [c for c in out.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        out["DOCUMENT_COUNT"] = out[doc_cols].sum(axis=1).astype(np.int8)

    contact_cols = [
        c
        for c in (
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        )
        if c in out.columns
    ]
    if contact_cols:
        out["CONTACT_COUNT"] = out[contact_cols].sum(axis=1).astype(np.int8)

    # --- Missingness as signal ----------------------------------------------
    # Incomplete applications are systematically riskier, independent of what
    # the missing fields would have said. Counted on the input frame so the
    # engineered columns above do not contaminate the count.
    out["MISSING_COUNT"] = df.isna().sum(axis=1).astype(np.int16)

    return out


def build_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean then enrich the application table. The public entry point."""
    return add_domain_features(clean_application(df))
