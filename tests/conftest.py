"""Shared fixtures.

Every fixture here is synthetic and tiny. The test suite must run in CI where
the 2.7 GB dataset does not exist, so nothing in ``tests/`` may read
``data/raw`` unless it is marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.application import DAYS_EMPLOYED_SENTINEL


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator, so a failing test fails identically on every run."""
    return np.random.default_rng(42)


@pytest.fixture
def application_df() -> pd.DataFrame:
    """A minimal application table exercising each data-quality edge case.

    Row 0: ordinary applicant.
    Row 1: sentinel ``DAYS_EMPLOYED`` (pensioner).
    Row 2: ``XNA`` gender, which must be dropped.
    Row 3: zero income, which must not produce an infinite ratio.
    Row 4: all external scores missing.
    """
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5],
            "TARGET": [0, 1, 0, 1, 0],
            "CODE_GENDER": ["M", "F", "XNA", "F", "M"],
            "AMT_INCOME_TOTAL": [100_000.0, 50_000.0, 80_000.0, 0.0, 120_000.0],
            "AMT_CREDIT": [500_000.0, 200_000.0, 300_000.0, 150_000.0, 600_000.0],
            "AMT_ANNUITY": [25_000.0, 12_000.0, 18_000.0, 9_000.0, 30_000.0],
            "AMT_GOODS_PRICE": [450_000.0, 180_000.0, 270_000.0, 140_000.0, 550_000.0],
            "DAYS_BIRTH": [-12_000, -20_000, -15_000, -10_000, -18_000],
            "DAYS_EMPLOYED": [-2_000, DAYS_EMPLOYED_SENTINEL, -3_000, -500, -6_000],
            "DAYS_REGISTRATION": [-4_000.0, -9_000.0, -5_000.0, -2_000.0, -7_000.0],
            "DAYS_ID_PUBLISH": [-3_000, -4_000, -3_500, -1_500, -5_000],
            "CNT_CHILDREN": [0, 2, 1, 3, 0],
            "CNT_FAM_MEMBERS": [2.0, 4.0, 3.0, 5.0, 1.0],
            "EXT_SOURCE_1": [0.5, 0.3, np.nan, 0.7, np.nan],
            "EXT_SOURCE_2": [0.6, 0.2, 0.4, np.nan, np.nan],
            "EXT_SOURCE_3": [0.7, np.nan, 0.5, 0.6, np.nan],
            "FLAG_DOCUMENT_2": [0, 1, 0, 0, 1],
            "FLAG_DOCUMENT_3": [1, 1, 1, 0, 1],
            "FLAG_PHONE": [1, 0, 1, 1, 0],
            "FLAG_EMAIL": [0, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def bureau_df() -> pd.DataFrame:
    """Bureau records for three applicants with contrasting credit files."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2, 2, 3],
            "SK_ID_BUREAU": [101, 102, 103, 201, 202, 301],
            "CREDIT_ACTIVE": [
                "Active",
                "Closed",
                "Closed",
                "Active",
                "Active",
                "Closed",
            ],
            "CREDIT_TYPE": [
                "Consumer credit",
                "Credit card",
                "Consumer credit",
                "Car loan",
                "Consumer credit",
                "Mortgage",
            ],
            "DAYS_CREDIT": [-100, -500, -900, -50, -200, -1500],
            "DAYS_CREDIT_ENDDATE": [200.0, -300.0, -700.0, 400.0, 100.0, -1000.0],
            "DAYS_CREDIT_UPDATE": [-10, -400, -800, -5, -150, -1400],
            "CREDIT_DAY_OVERDUE": [0, 0, 30, 0, 0, 0],
            "AMT_CREDIT_MAX_OVERDUE": [0.0, 0.0, 5_000.0, 0.0, 0.0, 0.0],
            "AMT_CREDIT_SUM": [50_000.0, 20_000.0, 30_000.0, 200_000.0, 40_000.0, 0.0],
            "AMT_CREDIT_SUM_DEBT": [25_000.0, 0.0, 0.0, 150_000.0, 10_000.0, 0.0],
            "AMT_CREDIT_SUM_LIMIT": [0.0, 20_000.0, 0.0, 0.0, 0.0, 0.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 5_000.0, 0.0, 0.0, 0.0],
            "AMT_ANNUITY": [5_000.0, 2_000.0, 3_000.0, 20_000.0, 4_000.0, 0.0],
            "CNT_CREDIT_PROLONG": [0, 0, 1, 0, 0, 0],
        }
    )


@pytest.fixture
def bureau_balance_df() -> pd.DataFrame:
    """Monthly status history for two of the bureau credits."""
    return pd.DataFrame(
        {
            "SK_ID_BUREAU": [101, 101, 101, 102, 102, 103, 103, 201],
            "MONTHS_BALANCE": [0, -1, -2, 0, -1, 0, -1, 0],
            "STATUS": ["0", "0", "1", "C", "C", "2", "1", "0"],
        }
    )


@pytest.fixture
def separable_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Perfectly separated scores: every metric should be at its maximum."""
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = np.concatenate([np.linspace(0.01, 0.4, 90), np.linspace(0.6, 0.99, 10)])
    return y_true, y_prob


@pytest.fixture
def realistic_predictions(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Imperfect but informative scores at a realistic 8% base rate."""
    n = 5_000
    y_true = rng.binomial(1, 0.08, size=n)
    # Positives are shifted upward but the distributions overlap heavily.
    noise = rng.normal(0, 1.0, size=n)
    logits = -2.5 + 1.5 * y_true + noise
    y_prob = 1 / (1 + np.exp(-logits))
    return y_true, y_prob
