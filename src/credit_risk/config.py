"""Central configuration: paths, column conventions and reproducibility constants.

Every module imports paths from here rather than hard-coding them, so the
pipeline runs identically on a laptop, in CI, or in a container.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset conventions
# ---------------------------------------------------------------------------
KAGGLE_COMPETITION = "home-credit-default-risk"

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
BUREAU_ID = "SK_ID_BUREAU"
PREV_ID = "SK_ID_PREV"

#: Raw CSVs shipped by the competition, keyed by the name we refer to them by.
RAW_TABLES: dict[str, str] = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_FOLDS = 5

# ---------------------------------------------------------------------------
# Business assumptions
# ---------------------------------------------------------------------------
# A rejected good customer costs us the profit we would have made on them;
# an accepted bad customer costs us the unrecovered principal. The asymmetry
# between these two numbers is what sets the decision threshold — not 0.5.
# These are stated assumptions, not facts from the data; see docs/methodology.md.
LGD = 0.65  # loss given default: fraction of exposure lost when a loan defaults
PROFIT_MARGIN = 0.12  # expected profit on a performing loan, as a fraction of credit
