"""Tests for the evaluation metrics.

These pin down the properties the metrics must have. Most are checked against
hand-computable cases rather than against a second implementation, so a test
failing means the metric is wrong, not merely different.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.evaluation.metrics import (
    evaluate,
    expected_calibration_error,
    ks_statistic,
    lift_at_k,
)


class TestKSStatistic:
    def test_perfect_separation_gives_one(self, separable_predictions):
        y_true, y_prob = separable_predictions
        ks, _ = ks_statistic(y_true, y_prob)
        assert ks == pytest.approx(1.0)

    def test_uninformative_scores_give_near_zero(self, rng):
        y_true = rng.binomial(1, 0.08, size=10_000)
        y_prob = rng.uniform(size=10_000)  # independent of the target
        ks, _ = ks_statistic(y_true, y_prob)
        assert ks < 0.05

    def test_threshold_lies_within_score_range(self, realistic_predictions):
        y_true, y_prob = realistic_predictions
        _, threshold = ks_statistic(y_true, y_prob)
        assert y_prob.min() <= threshold <= y_prob.max()

    def test_inverted_scores_still_separate(self, separable_predictions):
        """KS measures separation, so it is not symmetric under inversion.

        A perfectly *anti*-correlated score has max(TPR - FPR) of 0: the metric
        reports no useful separation in the direction the model claims, which
        is the correct answer for a model that ranks backwards.
        """
        y_true, y_prob = separable_predictions
        ks, _ = ks_statistic(y_true, 1 - y_prob)
        assert ks == pytest.approx(0.0, abs=1e-9)


class TestExpectedCalibrationError:
    def test_well_calibrated_model_scores_near_zero(self, rng):
        n = 50_000
        y_prob = rng.uniform(0.01, 0.5, size=n)
        y_true = rng.binomial(1, y_prob)  # outcomes generated *from* the scores
        assert expected_calibration_error(y_true, y_prob) < 0.02

    def test_systematically_overconfident_model_is_penalised(self, rng):
        n = 20_000
        true_p = rng.uniform(0.01, 0.3, size=n)
        y_true = rng.binomial(1, true_p)
        y_prob = np.clip(true_p * 3, 0, 1)  # predicts 3x the real risk
        assert expected_calibration_error(y_true, y_prob) > 0.1

    def test_constant_predictions_do_not_crash(self):
        """Degenerate input: quantile edges collapse to a single value."""
        y_true = np.array([0, 1, 0, 0, 1])
        y_prob = np.full(5, 0.4)
        assert expected_calibration_error(y_true, y_prob) >= 0.0

    def test_rejects_unknown_strategy(self, realistic_predictions):
        y_true, y_prob = realistic_predictions
        with pytest.raises(ValueError, match="strategy"):
            expected_calibration_error(y_true, y_prob, strategy="sqrt")


class TestLiftAtK:
    def test_all_positives_in_top_decile(self):
        """100 positives out of 1000, ranked first: lift is exactly 1/k."""
        y_true = np.array([1] * 100 + [0] * 900)
        y_prob = np.concatenate([np.full(100, 0.9), np.full(900, 0.1)])
        assert lift_at_k(y_true, y_prob, k=0.10) == pytest.approx(10.0)

    def test_random_ranking_gives_lift_near_one(self, rng):
        y_true = rng.binomial(1, 0.1, size=20_000)
        y_prob = rng.uniform(size=20_000)
        assert lift_at_k(y_true, y_prob, k=0.10) == pytest.approx(1.0, abs=0.2)

    def test_k_of_one_covers_whole_population(self, realistic_predictions):
        y_true, y_prob = realistic_predictions
        assert lift_at_k(y_true, y_prob, k=1.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad_k", [0.0, -0.1, 1.5])
    def test_rejects_k_outside_range(self, realistic_predictions, bad_k):
        y_true, y_prob = realistic_predictions
        with pytest.raises(ValueError, match="k must lie"):
            lift_at_k(y_true, y_prob, k=bad_k)


class TestEvaluate:
    def test_gini_is_derived_from_auc(self, realistic_predictions):
        report = evaluate(*realistic_predictions)
        assert report.gini == pytest.approx(2 * report.roc_auc - 1)

    def test_reports_the_base_rate(self, realistic_predictions):
        y_true, y_prob = realistic_predictions
        report = evaluate(y_true, y_prob)
        assert report.default_rate == pytest.approx(y_true.mean())
        assert report.n_samples == len(y_true)

    def test_informative_model_beats_chance(self, realistic_predictions):
        report = evaluate(*realistic_predictions)
        assert report.roc_auc > 0.6
        # PR-AUC must clear the base rate, which is the no-skill floor.
        assert report.pr_auc > report.default_rate

    def test_serialises_to_plain_types(self, realistic_predictions):
        report = evaluate(*realistic_predictions)
        as_dict = report.to_dict()
        assert set(as_dict) >= {"roc_auc", "pr_auc", "ks_statistic", "brier_score"}
        assert all(isinstance(v, (int, float)) for v in as_dict.values())


class TestInputValidation:
    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluate(np.array([0, 1, 0]), np.array([0.1, 0.9]))

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            evaluate(np.array([]), np.array([]))

    def test_rejects_non_binary_target(self):
        with pytest.raises(ValueError, match="binary"):
            evaluate(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))

    def test_rejects_probabilities_outside_unit_interval(self):
        """Catches raw log-odds being passed in place of probabilities."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            evaluate(np.array([0, 1]), np.array([-2.0, 3.5]))

    def test_rejects_nan_predictions(self):
        with pytest.raises(ValueError, match="NaN"):
            evaluate(np.array([0, 1]), np.array([0.5, np.nan]))

    def test_rejects_single_class_target(self):
        with pytest.raises(ValueError, match="both classes"):
            evaluate(np.zeros(10), np.linspace(0.1, 0.9, 10))
