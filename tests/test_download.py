"""Tests for credential discovery.

No test here touches the network. They cover the part that actually broke in
practice: the Kaggle CLI changed which file it reads between versions, and code
that hard-codes one location fails on a correctly configured machine while
insisting the credentials are missing.
"""

from __future__ import annotations

import pytest

from credit_risk.data import download as dl


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point credential lookup at an empty temp directory and clear the env."""
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()

    monkeypatch.setattr(
        dl,
        "CREDENTIAL_PATHS",
        (
            kaggle_dir / "credentials.json",
            kaggle_dir / "access_token",
            kaggle_dir / "access_token.txt",
            kaggle_dir / "kaggle.json",
        ),
    )
    for var in dl.CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return kaggle_dir


class TestFindCredentials:
    def test_returns_none_when_nothing_is_configured(self, isolated_home):
        assert dl.find_credentials() is None
        assert dl.credentials_available() is False

    @pytest.mark.parametrize(
        "filename",
        ["credentials.json", "access_token", "access_token.txt", "kaggle.json"],
    )
    def test_detects_each_supported_file(self, isolated_home, filename):
        """Every mechanism the CLI has shipped must be recognised."""
        (isolated_home / filename).write_text("{}")
        found = dl.find_credentials()
        assert found is not None
        assert found.name == filename
        assert dl.credentials_available() is True

    @pytest.mark.parametrize("var", ["KAGGLE_API_TOKEN", "KAGGLE_KEY"])
    def test_detects_environment_variables(self, isolated_home, monkeypatch, var):
        monkeypatch.setenv(var, "some-token-value")
        assert dl.find_credentials() == var
        assert dl.credentials_available() is True

    def test_ignores_empty_environment_variable(self, isolated_home, monkeypatch):
        """An exported-but-empty variable is not a credential."""
        monkeypatch.setenv("KAGGLE_API_TOKEN", "")
        assert dl.find_credentials() is None

    def test_prefers_oauth_over_legacy(self, isolated_home):
        """With several present, the newest mechanism wins."""
        (isolated_home / "kaggle.json").write_text("{}")
        (isolated_home / "credentials.json").write_text("{}")
        assert dl.find_credentials().name == "credentials.json"

    def test_a_directory_is_not_a_credential_file(self, isolated_home):
        (isolated_home / "kaggle.json").mkdir()
        assert dl.find_credentials() is None


class TestDownloadGuards:
    def test_missing_credentials_raises_with_actionable_help(self, isolated_home, monkeypatch):
        monkeypatch.setattr(dl, "already_downloaded", lambda: False)
        with pytest.raises(RuntimeError) as excinfo:
            dl.download()

        message = str(excinfo.value)
        assert "kaggle auth login" in message
        assert "rules" in message  # the most common real cause

    def test_skips_work_when_data_is_already_present(self, monkeypatch):
        monkeypatch.setattr(dl, "already_downloaded", lambda: True)

        def fail(*args, **kwargs):
            raise AssertionError("should not shell out when data is present")

        monkeypatch.setattr(dl.subprocess, "run", fail)
        assert dl.download() == dl.RAW_DIR
