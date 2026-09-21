"""Fetch the Home Credit Default Risk dataset from Kaggle.

The raw data is ~2.7 GB uncompressed and is deliberately git-ignored: the
repository stores the *code that reproduces* the data, not the data itself.

Authentication requires a Kaggle account and acceptance of the competition
rules at https://www.kaggle.com/competitions/home-credit-default-risk/rules

The Kaggle CLI has accepted credentials from several locations across its
versions, so this module checks all of them rather than assuming one. Newer
CLIs (>= 2.x) prefer OAuth via ``kaggle auth login``; older ones expect a
legacy ``kaggle.json``. Both still work, and a project that only checked for
one would fail confusingly on a machine set up the other way.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

from credit_risk.config import KAGGLE_COMPETITION, RAW_DIR, RAW_TABLES

KAGGLE_DIR = Path.home() / ".kaggle"

#: Credential files the CLI reads, newest mechanism first.
CREDENTIAL_PATHS: tuple[Path, ...] = (
    KAGGLE_DIR / "credentials.json",  # OAuth, written by `kaggle auth login`
    KAGGLE_DIR / "access_token",  # API token pasted from the settings page
    KAGGLE_DIR / "access_token.txt",
    KAGGLE_DIR / "kaggle.json",  # legacy username/key pair
)

#: Environment variables the CLI reads in place of a file.
CREDENTIAL_ENV_VARS: tuple[str, ...] = ("KAGGLE_API_TOKEN", "KAGGLE_KEY")

AUTH_HELP = """Kaggle credentials not found.

Easiest (no token to manage):
    python -m kaggle auth login

Or generate a token at https://www.kaggle.com/settings/api and either:
    export KAGGLE_API_TOKEN=<token>
    echo '<token>' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

Either way, accept the competition rules first:
    https://www.kaggle.com/competitions/{competition}/rules"""


def find_credentials() -> Path | str | None:
    """Return the credential source the CLI will use, or None if there is none.

    Returns:
        A path to the credential file, the name of the environment variable
        supplying the token, or None.
    """
    for path in CREDENTIAL_PATHS:
        if path.is_file():
            return path
    for var in CREDENTIAL_ENV_VARS:
        if os.environ.get(var):
            return var
    return None


def credentials_available() -> bool:
    """Return True if any supported credential source is present."""
    return find_credentials() is not None


def already_downloaded() -> bool:
    """Return True if every expected raw CSV is already on disk."""
    return all((RAW_DIR / fname).is_file() for fname in RAW_TABLES.values())


def download(force: bool = False) -> Path:
    """Download and extract the competition archive into ``data/raw``.

    Args:
        force: Re-download even if the CSVs are already present.

    Returns:
        The directory containing the extracted CSVs.

    Raises:
        RuntimeError: If credentials are missing, or the download or
            extraction fails.
    """
    if already_downloaded() and not force:
        print(f"Raw data already present in {RAW_DIR} — nothing to do.")
        return RAW_DIR

    source = find_credentials()
    if source is None:
        raise RuntimeError(AUTH_HELP.format(competition=KAGGLE_COMPETITION))

    print(f"Authenticating via {source}")
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
        # By far the most common cause is un-accepted competition rules, which
        # the API reports as a generic 403 rather than anything actionable.
        raise RuntimeError(
            "Kaggle download failed. The usual cause is that the competition "
            "rules have not been accepted on your account. Visit\n"
            f"  https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules\n"
            "and click 'I Understand and Accept', then retry.\n\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    archive = RAW_DIR / f"{KAGGLE_COMPETITION}.zip"
    if not archive.is_file():
        candidates = sorted(RAW_DIR.glob("*.zip"))
        if not candidates:
            raise RuntimeError(f"Download reported success but no archive in {RAW_DIR}")
        archive = candidates[0]

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
