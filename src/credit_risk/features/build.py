"""Assemble the modelling table from the raw tables.

Run as ``make features``. Reads ``data/raw``, writes ``data/processed``.

The output is a single Parquet file, one row per applicant, with the target in
a ``TARGET`` column and the applicant id preserved so that predictions can be
traced back. Building it is deterministic: the same raw inputs always produce
the same table, which is what makes a model result reproducible.
"""

from __future__ import annotations

import time

import pandas as pd

from credit_risk.config import ID_COL, PROCESSED_DIR, TARGET
from credit_risk.data.load import load_table, reduce_memory_usage
from credit_risk.features.application import build_application_features
from credit_risk.features.bureau import build_bureau_features


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns to pandas ``category`` dtype.

    LightGBM consumes the category dtype directly and finds optimal splits over
    category subsets. That beats one-hot encoding on this data: several columns
    (``ORGANIZATION_TYPE``, 58 levels) would explode the feature count, and it
    beats ordinal encoding, which invents a false ordering between levels.

    Columns are selected by what they are *not* rather than by asking for
    ``object``. Pandas 3 gives strings their own dtype, so ``select_dtypes
    (include="object")`` is deprecated and its meaning differs between 2.x and
    3.x; this form behaves identically on both.
    """
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            continue
        if (
            pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_datetime64_any_dtype(dtype)
        ):
            continue
        out[col] = out[col].astype("category")
    return out


def build_feature_table(
    split: str = "train", include_bureau: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """Build the full modelling table for one split.

    Args:
        split: ``"train"`` or ``"test"``.
        include_bureau: Join the aggregated bureau features.
        verbose: Log each stage with timings and shapes.

    Returns:
        The assembled table.

    Raises:
        ValueError: If ``split`` is not "train" or "test".
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    t0 = time.perf_counter()

    app = load_table(f"application_{split}")
    if verbose:
        print(f"application_{split:<5} {app.shape[0]:>7,} x {app.shape[1]:>3}")

    df = build_application_features(app)
    if verbose:
        print(f"  + domain features  -> {df.shape[1]:>3} columns")

    if include_bureau:
        bureau = load_table("bureau")
        bureau_balance = load_table("bureau_balance")
        bureau_features = build_bureau_features(bureau, bureau_balance)

        df = df.merge(bureau_features, how="left", left_on=ID_COL, right_index=True)

        # A left join leaves NaN for applicants with no bureau file at all.
        # That is meaningful — a thin file is not the same as a clean one — so
        # flag it rather than imputing it away.
        df["HAS_BUREAU_HISTORY"] = df["BUREAU_CREDIT_COUNT"].notna().astype("int8")
        df["BUREAU_CREDIT_COUNT"] = df["BUREAU_CREDIT_COUNT"].fillna(0)

        if verbose:
            print(f"  + bureau features  -> {df.shape[1]:>3} columns")

    df = encode_categoricals(df)
    df = reduce_memory_usage(df)

    if verbose:
        elapsed = time.perf_counter() - t0
        mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f"  final {df.shape[0]:,} x {df.shape[1]} | {mem:,.0f} MB | {elapsed:.1f}s")
        if TARGET in df.columns:
            print(f"  default rate {df[TARGET].mean():.4%}")

    return df


def main() -> None:
    """Build and persist both splits."""
    for split in ("train", "test"):
        df = build_feature_table(split=split)
        out_path = PROCESSED_DIR / f"{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  wrote {out_path.relative_to(PROCESSED_DIR.parents[1])}\n")


if __name__ == "__main__":
    main()
