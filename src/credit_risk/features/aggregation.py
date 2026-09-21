"""Reusable helpers for collapsing one-to-many tables onto the application key.

Six of the seven Home Credit tables are *child* tables: one applicant has many
bureau records, many previous applications, many instalments. A model needs one
row per applicant, so each child table has to be collapsed by ``SK_ID_CURR``.

How that collapse is done is most of the modelling work in this project. The
helpers here exist so that each table's module can describe *what* to aggregate
and stay out of the mechanics of *how*.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Flatten the MultiIndex that ``groupby().agg()`` produces.

    ``agg({"AMT_CREDIT_SUM": ["mean", "max"]})`` yields column tuples like
    ``("AMT_CREDIT_SUM", "mean")``. Flattened and prefixed, that becomes
    ``BUREAU_AMT_CREDIT_SUM_MEAN`` — unambiguous once several tables are joined
    side by side, which matters when reading a feature-importance plot.

    Args:
        df: Frame whose columns are a 2-level MultiIndex.
        prefix: Table identifier prepended to every name, e.g. ``"BUREAU"``.

    Returns:
        The same frame with flat, uppercase, prefixed column names.
    """
    df = df.copy()
    df.columns = pd.Index(
        [f"{prefix}_{col[0]}_{col[1]}".upper() for col in df.columns.to_flat_index()]
    )
    return df


def aggregate_by_key(
    df: pd.DataFrame,
    key: str,
    aggregations: dict[str, Sequence[str]],
    prefix: str,
) -> pd.DataFrame:
    """Group a child table by ``key`` and apply per-column aggregations.

    Args:
        df: The child table.
        key: Column to group on, normally ``SK_ID_CURR``.
        aggregations: Column name -> list of aggregation functions.
        prefix: Prefix for the resulting column names.

    Returns:
        One row per ``key``, indexed by it.

    Raises:
        KeyError: If ``key`` or any aggregated column is absent.
    """
    if key not in df.columns:
        raise KeyError(f"Key column {key!r} not found in frame")

    missing = set(aggregations) - set(df.columns)
    if missing:
        raise KeyError(f"Columns not found in frame: {sorted(missing)}")

    grouped = df.groupby(key).agg({col: list(fns) for col, fns in aggregations.items()})
    return flatten_columns(grouped, prefix)


def one_hot_counts(
    df: pd.DataFrame,
    key: str,
    column: str,
    prefix: str,
    normalise: bool = True,
) -> pd.DataFrame:
    """Count how often each category of ``column`` appears per ``key``.

    For a categorical child column such as ``CREDIT_ACTIVE`` this produces one
    feature per category — "how many of this applicant's bureau credits are
    Active?" — which is exactly the kind of question the raw table cannot
    answer in a single row.

    Args:
        df: The child table.
        key: Grouping key.
        column: Categorical column to count.
        prefix: Prefix for resulting column names.
        normalise: Also emit each count as a fraction of the applicant's total,
            so that an applicant with 2 of 2 active credits is distinguishable
            from one with 2 of 40.

    Returns:
        One row per ``key``, indexed by it.
    """
    counts = (
        pd.crosstab(df[key], df[column])
        .add_prefix(f"{prefix}_{column}_".upper())
        .add_suffix("_COUNT")
    )
    counts.columns = pd.Index([c.upper().replace(" ", "_") for c in counts.columns])

    if not normalise:
        return counts

    totals = counts.sum(axis=1).replace(0, pd.NA)
    fractions = counts.div(totals, axis=0)
    fractions.columns = pd.Index([c.replace("_COUNT", "_FRAC") for c in fractions.columns])

    return counts.join(fractions)
