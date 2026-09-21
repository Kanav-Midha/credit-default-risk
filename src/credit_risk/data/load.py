"""Loading raw tables with aggressive but lossless memory reduction.

The raw CSVs occupy ~2.7 GB because pandas defaults every integer to int64 and
every float to float64. Most columns need a fraction of that range. Downcasting
to the narrowest dtype that holds the observed values typically cuts memory by
60-70%, which is the difference between this pipeline running comfortably on a
laptop and thrashing swap.

Loaded tables are cached as Parquet in ``data/interim``. Parquet stores dtypes
in its schema, so the expensive CSV parse and downcast happen exactly once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import INTERIM_DIR, RAW_DIR, RAW_TABLES


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Downcast numeric columns to the narrowest dtype that holds their values.

    Integers are downcast within the signed/unsigned integer families. Floats
    are downcast to float32 only — never float16, whose ~3 decimal digits of
    precision silently corrupt monetary columns such as ``AMT_CREDIT``.

    Columns containing NaN are left as floats, since pandas' nullable integer
    dtypes are not uniformly supported by the downstream gradient-boosting
    libraries.

    Args:
        df: Frame to compress. Not modified in place.
        verbose: Print the before/after footprint.

    Returns:
        A new frame with identical values and narrower dtypes.
    """
    out = df.copy()
    start_mb = out.memory_usage(deep=True).sum() / 1024**2

    for col in out.columns:
        col_type = out[col].dtype

        # Only numeric columns can be downcast. Testing for object dtype is
        # not enough on pandas 3, where strings have a dtype of their own.
        if not pd.api.types.is_numeric_dtype(col_type) or pd.api.types.is_bool_dtype(col_type):
            continue

        c_min, c_max = out[col].min(), out[col].max()
        if pd.isna(c_min) or pd.isna(c_max):
            continue  # all-NaN column: nothing to infer from

        if pd.api.types.is_integer_dtype(col_type):
            if c_min >= 0:
                for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
                    if c_max <= np.iinfo(dtype).max:
                        out[col] = out[col].astype(dtype)
                        break
            else:
                for dtype in (np.int8, np.int16, np.int32, np.int64):
                    if c_min >= np.iinfo(dtype).min and c_max <= np.iinfo(dtype).max:
                        out[col] = out[col].astype(dtype)
                        break
        # float32 holds ~7 significant digits: ample for every column here, and
        # it keeps integers exactly representable up to 2^24.
        elif pd.api.types.is_float_dtype(col_type) and (
            c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max
        ):
            out[col] = out[col].astype(np.float32)

    if verbose:
        end_mb = out.memory_usage(deep=True).sum() / 1024**2
        saved = 100 * (start_mb - end_mb) / start_mb if start_mb else 0.0
        print(f"  memory {start_mb:,.1f} MB -> {end_mb:,.1f} MB ({saved:.1f}% saved)")

    return out


def load_table(name: str, use_cache: bool = True, verbose: bool = False) -> pd.DataFrame:
    """Load one raw table by logical name, via a Parquet cache.

    Args:
        name: Key from :data:`credit_risk.config.RAW_TABLES`, e.g. ``"bureau"``.
        use_cache: Read from / write to the Parquet cache in ``data/interim``.
        verbose: Print shape and memory savings.

    Returns:
        The table as a memory-reduced DataFrame.

    Raises:
        KeyError: If ``name`` is not a known table.
        FileNotFoundError: If the raw CSV has not been downloaded yet.
    """
    if name not in RAW_TABLES:
        raise KeyError(f"Unknown table {name!r}. Expected one of {sorted(RAW_TABLES)}")

    cache_path = INTERIM_DIR / f"{name}.parquet"
    if use_cache and cache_path.is_file():
        df = pd.read_parquet(cache_path)
        if verbose:
            print(f"{name}: {df.shape[0]:,} x {df.shape[1]} (from cache)")
        return df

    csv_path = RAW_DIR / RAW_TABLES[name]
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `make data` to download the dataset first."
        )

    if verbose:
        print(f"{name}: parsing {csv_path.name}...")
    df = pd.read_csv(csv_path)
    df = reduce_memory_usage(df, verbose=verbose)

    if use_cache:
        df.to_parquet(cache_path, index=False)

    if verbose:
        print(f"{name}: {df.shape[0]:,} x {df.shape[1]}")
    return df


def load_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Load every raw table into a dict keyed by logical name."""
    return {name: load_table(name, verbose=verbose) for name in RAW_TABLES}


if __name__ == "__main__":
    for table_name in RAW_TABLES:
        load_table(table_name, verbose=True)
