"""Tests for feature engineering.

The data-quality tests matter most. A silent sentinel or an infinity leaking
into the feature matrix does not raise; it just quietly degrades the model, and
only a test like these catches it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.aggregation import aggregate_by_key, one_hot_counts
from credit_risk.features.application import (
    DAYS_EMPLOYED_SENTINEL,
    add_domain_features,
    build_application_features,
    clean_application,
)
from credit_risk.features.build import encode_categoricals
from credit_risk.features.bureau import aggregate_bureau_balance, build_bureau_features


class TestCleanApplication:
    def test_sentinel_employment_becomes_missing(self, application_df):
        cleaned = clean_application(application_df)
        assert not (cleaned["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).any()
        assert cleaned["DAYS_EMPLOYED"].isna().sum() == 1

    def test_sentinel_is_flagged_before_being_nulled(self, application_df):
        """The flag is the point: 'unemployed' differs from 'unknown tenure'."""
        cleaned = clean_application(application_df)
        flagged = cleaned.loc[cleaned["SK_ID_CURR"] == 2, "DAYS_EMPLOYED_IS_SENTINEL"]
        assert flagged.iloc[0] == 1
        assert cleaned["DAYS_EMPLOYED_IS_SENTINEL"].sum() == 1

    def test_xna_gender_rows_are_dropped(self, application_df):
        cleaned = clean_application(application_df)
        assert len(cleaned) == len(application_df) - 1
        assert "XNA" not in set(cleaned["CODE_GENDER"])

    def test_non_positive_income_becomes_missing(self, application_df):
        cleaned = clean_application(application_df)
        zero_income_row = cleaned.loc[cleaned["SK_ID_CURR"] == 4]
        assert zero_income_row["AMT_INCOME_TOTAL"].isna().all()

    def test_does_not_mutate_the_input(self, application_df):
        before = application_df.copy()
        clean_application(application_df)
        pd.testing.assert_frame_equal(application_df, before)


class TestDomainFeatures:
    def test_credit_income_ratio_is_correct(self, application_df):
        out = add_domain_features(application_df)
        row = out.loc[out["SK_ID_CURR"] == 1].iloc[0]
        assert row["CREDIT_INCOME_RATIO"] == pytest.approx(500_000 / 100_000)

    def test_age_is_converted_from_negative_days(self, application_df):
        out = add_domain_features(application_df)
        row = out.loc[out["SK_ID_CURR"] == 1].iloc[0]
        assert row["AGE_YEARS"] == pytest.approx(12_000 / 365.25)

    def test_division_by_zero_yields_nan_not_infinity(self, application_df):
        """Infinity is worse than missing: LightGBM handles NaN, not inf."""
        out = add_domain_features(application_df)
        numeric = out.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).any()

        zero_income = out.loc[out["SK_ID_CURR"] == 4].iloc[0]
        assert pd.isna(zero_income["CREDIT_INCOME_RATIO"])

    def test_external_source_summary_ignores_missing_values(self, application_df):
        out = add_domain_features(application_df)
        row = out.loc[out["SK_ID_CURR"] == 1].iloc[0]
        assert row["EXT_SOURCE_MEAN"] == pytest.approx(np.mean([0.5, 0.6, 0.7]))
        assert row["EXT_SOURCE_MIN"] == pytest.approx(0.5)
        assert row["EXT_SOURCE_NA_COUNT"] == 0

    def test_counts_missing_external_sources(self, application_df):
        out = add_domain_features(application_df)
        row = out.loc[out["SK_ID_CURR"] == 5].iloc[0]
        assert row["EXT_SOURCE_NA_COUNT"] == 3
        assert pd.isna(row["EXT_SOURCE_MEAN"])

    def test_document_and_contact_counts(self, application_df):
        out = add_domain_features(application_df)
        row = out.loc[out["SK_ID_CURR"] == 1].iloc[0]
        assert row["DOCUMENT_COUNT"] == 1  # FLAG_DOCUMENT_2=0, _3=1
        assert row["CONTACT_COUNT"] == 1  # FLAG_PHONE=1, FLAG_EMAIL=0

    def test_adds_columns_without_removing_any(self, application_df):
        out = add_domain_features(application_df)
        assert set(application_df.columns).issubset(out.columns)
        assert out.shape[1] > application_df.shape[1]


class TestBuildApplicationFeatures:
    def test_pipeline_cleans_then_enriches(self, application_df):
        out = build_application_features(application_df)
        assert len(out) == len(application_df) - 1  # XNA dropped
        assert "CREDIT_INCOME_RATIO" in out.columns
        assert "DAYS_EMPLOYED_IS_SENTINEL" in out.columns
        assert not (out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).any()

    def test_is_deterministic(self, application_df):
        pd.testing.assert_frame_equal(
            build_application_features(application_df),
            build_application_features(application_df),
        )


class TestAggregationHelpers:
    def test_aggregate_by_key_flattens_and_prefixes(self, bureau_df):
        out = aggregate_by_key(
            bureau_df,
            key="SK_ID_CURR",
            aggregations={"AMT_CREDIT_SUM": ["sum"]},
            prefix="BUREAU",
        )
        assert list(out.columns) == ["BUREAU_AMT_CREDIT_SUM_SUM"]
        assert out.loc[1, "BUREAU_AMT_CREDIT_SUM_SUM"] == pytest.approx(100_000.0)

    def test_aggregate_by_key_rejects_unknown_columns(self, bureau_df):
        with pytest.raises(KeyError, match="not found"):
            aggregate_by_key(
                bureau_df,
                key="SK_ID_CURR",
                aggregations={"NOPE": ["sum"]},
                prefix="BUREAU",
            )

    def test_one_hot_counts_produces_counts_and_fractions(self, bureau_df):
        out = one_hot_counts(bureau_df, "SK_ID_CURR", "CREDIT_ACTIVE", "BUREAU")
        assert out.loc[1, "BUREAU_CREDIT_ACTIVE_ACTIVE_COUNT"] == 1
        assert out.loc[1, "BUREAU_CREDIT_ACTIVE_CLOSED_COUNT"] == 2
        # Applicant 1 has 3 credits, 1 active.
        assert out.loc[1, "BUREAU_CREDIT_ACTIVE_ACTIVE_FRAC"] == pytest.approx(1 / 3)


class TestBureauFeatures:
    def test_balance_aggregation_counts_delinquent_months(self, bureau_balance_df):
        out = aggregate_bureau_balance(bureau_balance_df)
        # Credit 101: statuses 0, 0, 1 -> one delinquent month of three.
        assert out.loc[101, "BB_MONTHS_COUNT"] == 3
        assert out.loc[101, "BB_DPD_MONTHS"] == 1
        assert out.loc[101, "BB_DPD_RATIO"] == pytest.approx(1 / 3)
        # Credit 103: statuses 2, 1 -> both delinquent.
        assert out.loc[103, "BB_DPD_RATIO"] == pytest.approx(1.0)

    def test_one_row_per_applicant(self, bureau_df, bureau_balance_df):
        out = build_bureau_features(bureau_df, bureau_balance_df)
        assert len(out) == bureau_df["SK_ID_CURR"].nunique()
        assert out.index.is_unique

    def test_credit_count_matches_the_raw_table(self, bureau_df):
        out = build_bureau_features(bureau_df)
        assert out.loc[1, "BUREAU_CREDIT_COUNT"] == 3
        assert out.loc[3, "BUREAU_CREDIT_COUNT"] == 1

    def test_debt_to_credit_ratio(self, bureau_df):
        out = build_bureau_features(bureau_df)
        # Applicant 2: debt 160,000 against credit 240,000.
        assert out.loc[2, "BUREAU_TOTAL_DEBT_CREDIT_RATIO"] == pytest.approx(160_000 / 240_000)

    def test_zero_credit_does_not_produce_infinity(self, bureau_df):
        """Applicant 3's only credit has AMT_CREDIT_SUM of zero."""
        out = build_bureau_features(bureau_df)
        numeric = out.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).any()

    def test_works_without_the_balance_table(self, bureau_df):
        out = build_bureau_features(bureau_df, bureau_balance=None)
        assert "BUREAU_CREDIT_COUNT" in out.columns
        assert not any(c.startswith("BUREAU_BB_") for c in out.columns)


class TestEncodeCategoricals:
    """Pandas 3 gave strings their own dtype, so selecting by `object` no
    longer means what it used to. These pin the behaviour across versions."""

    def test_text_columns_become_categorical(self):
        df = pd.DataFrame({"name": ["a", "b", "a"], "n": [1, 2, 3]})
        out = encode_categoricals(df)
        assert isinstance(out["name"].dtype, pd.CategoricalDtype)

    def test_numeric_bool_and_datetime_are_left_alone(self):
        df = pd.DataFrame(
            {
                "i": [1, 2],
                "f": [1.5, 2.5],
                "b": [True, False],
                "t": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        )
        out = encode_categoricals(df)
        for col in df.columns:
            assert not isinstance(out[col].dtype, pd.CategoricalDtype), col

    def test_already_categorical_columns_are_untouched(self):
        df = pd.DataFrame({"c": pd.Categorical(["x", "y"], categories=["x", "y", "z"])})
        out = encode_categoricals(df)
        assert list(out["c"].cat.categories) == ["x", "y", "z"]

    def test_emits_no_deprecation_warnings(self):
        """The failure this guards against is a warning, not an exception."""
        df = pd.DataFrame({"name": ["a", "b"], "n": [1, 2]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            encode_categoricals(df)

    def test_values_survive_the_conversion(self):
        df = pd.DataFrame({"name": ["a", "b", "a"]})
        out = encode_categoricals(df)
        assert list(out["name"].astype(str)) == ["a", "b", "a"]
