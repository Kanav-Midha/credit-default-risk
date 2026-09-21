"""Tests for the cost model and threshold search.

The headline resume number comes out of this module, so its arithmetic is
pinned to hand-computed values rather than to whatever the code happens to
return.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.evaluation.business import (
    baseline_cost,
    expected_cost,
    optimal_threshold,
    sensitivity_analysis,
)


@pytest.fixture
def four_applicants():
    """Two defaulters and two good customers, equal exposure.

    At a 0.5 cutoff, applicants 0 and 3 are declined:
      - applicant 0 (prob 0.9, defaulted)  -> correctly declined, costs nothing
      - applicant 1 (prob 0.1, good)       -> approved, earns profit, costs nothing
      - applicant 2 (prob 0.2, defaulted)  -> approved and defaults -> lgd * 1000
      - applicant 3 (prob 0.8, good)       -> wrongly declined  -> margin * 1000
    """
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.2, 0.8])
    exposure = np.full(4, 1_000.0)
    return y_true, y_prob, exposure


class TestExpectedCost:
    def test_matches_hand_computation(self, four_applicants):
        y_true, y_prob, exposure = four_applicants
        cost = expected_cost(y_true, y_prob, exposure, threshold=0.5, lgd=0.65, margin=0.12)
        assert cost == pytest.approx(0.65 * 1_000 + 0.12 * 1_000)

    def test_threshold_of_one_approves_everyone(self, four_applicants):
        """Approving all means no forgone profit, only realised defaults."""
        y_true, y_prob, exposure = four_applicants
        cost = expected_cost(y_true, y_prob, exposure, threshold=1.01, lgd=0.65)
        assert cost == pytest.approx(2 * 0.65 * 1_000)

    def test_threshold_of_zero_declines_everyone(self, four_applicants):
        """Declining all means no credit losses, only forgone profit."""
        y_true, y_prob, exposure = four_applicants
        cost = expected_cost(y_true, y_prob, exposure, threshold=0.0, lgd=0.65, margin=0.12)
        assert cost == pytest.approx(2 * 0.12 * 1_000)

    def test_exposure_scales_cost_linearly(self, four_applicants):
        y_true, y_prob, exposure = four_applicants
        single = expected_cost(y_true, y_prob, exposure, 0.5)
        doubled = expected_cost(y_true, y_prob, exposure * 2, 0.5)
        assert doubled == pytest.approx(2 * single)

    def test_rejects_mismatched_shapes(self, four_applicants):
        y_true, y_prob, _ = four_applicants
        with pytest.raises(ValueError, match="Shape mismatch"):
            expected_cost(y_true, y_prob, np.ones(3), 0.5)


class TestBaselineCost:
    def test_counts_only_defaulters(self, four_applicants):
        y_true, _, exposure = four_applicants
        assert baseline_cost(y_true, exposure, lgd=0.65) == pytest.approx(1_300.0)

    def test_is_zero_when_nobody_defaults(self):
        assert baseline_cost(np.zeros(5), np.full(5, 1_000.0)) == pytest.approx(0.0)


class TestOptimalThreshold:
    def test_beats_the_naive_half_cutoff(self, realistic_predictions, rng):
        """The whole point of the module: 0.5 is not the right cutoff."""
        y_true, y_prob = realistic_predictions
        exposure = rng.uniform(50_000, 500_000, size=len(y_true))

        policy = optimal_threshold(y_true, y_prob, exposure)
        cost_at_half = expected_cost(y_true, y_prob, exposure, threshold=0.5)

        assert policy.total_cost <= cost_at_half
        assert policy.threshold < 0.5

    def test_beats_approving_everyone(self, realistic_predictions, rng):
        y_true, y_prob = realistic_predictions
        exposure = rng.uniform(50_000, 500_000, size=len(y_true))

        policy = optimal_threshold(y_true, y_prob, exposure)
        assert policy.total_cost < baseline_cost(y_true, exposure)

    def test_harsher_loss_assumption_tightens_the_cutoff(self, realistic_predictions, rng):
        """Raising LGD relative to margin must make lending more conservative."""
        y_true, y_prob = realistic_predictions
        exposure = rng.uniform(50_000, 500_000, size=len(y_true))

        lenient = optimal_threshold(y_true, y_prob, exposure, lgd=0.30, margin=0.20)
        strict = optimal_threshold(y_true, y_prob, exposure, lgd=0.90, margin=0.05)

        assert strict.threshold <= lenient.threshold
        assert strict.approval_rate <= lenient.approval_rate

    def test_policy_fields_are_internally_consistent(self, realistic_predictions, rng):
        y_true, y_prob = realistic_predictions
        exposure = rng.uniform(50_000, 500_000, size=len(y_true))
        policy = optimal_threshold(y_true, y_prob, exposure)

        assert 0.0 <= policy.approval_rate <= 1.0
        assert 0.0 <= policy.defaults_caught <= 1.0
        assert 0.0 <= policy.good_customers_rejected <= 1.0
        assert policy.cost_per_applicant == pytest.approx(policy.total_cost / len(y_true))
        # Screening out the riskiest applicants must improve the book's quality.
        assert policy.bad_rate_of_approved < y_true.mean()


class TestSensitivityAnalysis:
    def test_covers_the_full_grid(self, realistic_predictions, rng):
        y_true, y_prob = realistic_predictions
        exposure = rng.uniform(50_000, 500_000, size=len(y_true))

        records = sensitivity_analysis(
            y_true, y_prob, exposure, lgd_grid=(0.5, 0.7), margin_grid=(0.1, 0.15)
        )
        assert len(records) == 4
        assert {r["lgd"] for r in records} == {0.5, 0.7}

    def test_savings_stay_positive_across_assumptions(self, realistic_predictions, rng):
        """If this fails, the headline saving is an artefact of one guess."""
        y_true, y_prob = realistic_predictions
        exposure = rng.uniform(50_000, 500_000, size=len(y_true))

        records = sensitivity_analysis(y_true, y_prob, exposure)
        assert all(r["savings"] > 0 for r in records)
