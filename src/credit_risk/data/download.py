"""Fetch the Home Credit Default Risk dataset from Kaggle.

The raw data is ~2.7 GB uncompressed and is deliberately git-ignored: the
repository stores the *code that reproduces* the data, not the data itself.

Requires Kaggle API credentials at ``~/.kaggle/kaggle.json`` and acceptance of
the competition rules at
https://www.kaggle.com/competitions/home-credit-default-risk/rules
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from credit_risk.config import KAGGLE_COMPETITION, RAW_DIR, RAW_TABLES

CREDENTIALS_PATH = Path.home() / ".kaggle" / "kaggle.json"


def credentials_available() -> bool:
    """Return True if Kaggle API credentials are present and readable."""
    return CREDENTIALS_PATH.is_file()


def already_downloaded() -> bool:
    """Return True if every expected raw CSV is already on disk."""
    return all((RAW_DIR / fname).is_file() for fname in RAW_TABLES.values())


def download(force: bool = False) -> Path:
    """Download and extract the competition archive into ``data/raw``.

    Args:
        force: Re-download even if the CSVs are already present.

    Returns:
        The directory containing the extracted CSVs.
    """
    if already_downloaded() and not force:
        print(f"Raw data already present in {RAW_DIR} — nothing to do.")
        return RAW_DIR

    if not credentials_available():
        raise RuntimeError(
            f"Kaggle credentials not found at {CREDENTIALS_PATH}.\n"
            "Create an API token at https://www.kaggle.com/settings -> API -> "
            "'Create New Token', then move the downloaded kaggle.json to "
            f"{CREDENTIALS_PATH} and run: chmod 600 {CREDENTIALS_PATH}"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{KAGGLE_COMPETITION}' to {RAW_DIR} (~690 MB compressed)...")

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            KAGGLE_COMPETITION,
            "-p",
            str(RAW_DIR),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle download failed. Most commonly this means the competition "
            "rules have not been accepted on your account. Visit\n"
            f"  https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules\n"
            "and click 'I Understand and Accept', then retry.\n\n"
            f"stderr:\n{result.stderr}"
        )

    archive = RAW_DIR / f"{KAGGLE_COMPETITION}.zip"
    print(f"Extracting {archive.name}...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW_DIR)
    archive.unlink()

    missing = [f for f in RAW_TABLES.values() if not (RAW_DIR / f).is_file()]
    if missing:
        raise RuntimeError(f"Extraction incomplete; missing files: {missing}")

    total_mb = sum(p.stat().st_size for p in RAW_DIR.glob("*.csv")) / 1024**2
    print(f"Done. {len(RAW_TABLES)} tables, {total_mb:,.0f} MB in {RAW_DIR}")
    return RAW_DIR


if __name__ == "__main__":
    download(force="--force" in sys.argv)
