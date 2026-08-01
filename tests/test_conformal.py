"""Tests for conformal escalation-threshold calibration (FR-7).

The headline property (FR-7.3 / SC2): on exchangeable held-out data, the leaked-error
rate ``P(wrong AND confidence >= tau)`` stays at or below the nominal ``alpha``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from semopt.cascade.conformal import (
    calibrate_escalation_threshold,
    density_ratio_weights,
)


def _synth(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate (confidence, correct) where confidence is informative of correctness.

    Latent quality q ~ U(0,1); the row is correct w.p. q; reported confidence is q
    blurred by noise and clipped to [0,1]. Higher confidence ⇒ more likely correct, as a
    real calibrated-ish model behaves.
    """
    q = rng.uniform(0, 1, size=n)
    correct = rng.uniform(0, 1, size=n) < q
    confidence = np.clip(q + rng.normal(0, 0.1, size=n), 0.0, 1.0)
    return confidence, correct


def test_no_errors_accepts_everything():
    conf = np.array([0.5, 0.6, 0.7])
    correct = np.array([True, True, True])
    cal = calibrate_escalation_threshold(conf, correct, alpha=0.1)
    assert cal.tau == 0.0
    assert cal.accept_rate == 1.0
    assert cal.n_errors == 0


def test_all_errors_high_confidence_forces_escalation():
    # Every point is wrong yet highly confident: to keep leaked error low, tau must be
    # high enough that few/none are accepted.
    conf = np.full(100, 0.9)
    correct = np.zeros(100, dtype=bool)
    cal = calibrate_escalation_threshold(conf, correct, alpha=0.1)
    # Leaked weight must respect the budget.
    assert cal.leaked_error_weight <= 0.1 + 1e-9


def test_higher_alpha_accepts_more():
    rng = np.random.default_rng(0)
    conf, correct = _synth(rng, 2000)
    cal_lo = calibrate_escalation_threshold(conf, correct, alpha=0.05)
    cal_hi = calibrate_escalation_threshold(conf, correct, alpha=0.20)
    # A looser error target permits a lower threshold and thus more acceptance.
    assert cal_hi.tau <= cal_lo.tau
    assert cal_hi.accept_rate >= cal_lo.accept_rate


def test_invalid_alpha_raises():
    conf = np.array([0.5])
    correct = np.array([False])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            calibrate_escalation_threshold(conf, correct, alpha=bad)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        calibrate_escalation_threshold(np.array([0.5, 0.6]), np.array([True]), alpha=0.1)


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_empirical_coverage_holds_on_holdout(alpha: float):
    """Averaged over many exchangeable cal/test splits, leaked error <= alpha (SC2)."""
    leaked_rates = []
    for trial in range(40):
        rng = np.random.default_rng(1000 + trial)
        conf_cal, correct_cal = _synth(rng, 1500)
        conf_test, correct_test = _synth(rng, 1500)

        cal = calibrate_escalation_threshold(conf_cal, correct_cal, alpha)
        accepted = conf_test >= cal.tau
        leaked = np.mean(accepted & ~correct_test)
        leaked_rates.append(leaked)

    mean_leaked = float(np.mean(leaked_rates))
    # The guarantee is marginal; the mean over trials must not exceed alpha (small MC slack).
    assert mean_leaked <= alpha + 0.01, f"mean leaked {mean_leaked:.4f} > alpha {alpha}"


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 10_000), alpha=st.sampled_from([0.05, 0.10, 0.20]))
def test_property_leaked_error_bounded(seed: int, alpha: float):
    """Property: the leaked error is bounded by alpha *marginally over cal and test*.

    The guarantee is P(wrong ∧ accept) ≤ α in expectation over the calibration draw too —
    a single unlucky cal set can leak a little more on test. So we resample *both* cal and
    test each iteration and check the mean, which is what the theorem actually bounds.
    """
    base = int(seed)
    leaked_draws = []
    for i in range(12):
        rng = np.random.default_rng(base * 100 + i)
        conf_cal, correct_cal = _synth(rng, 2000)
        conf_test, correct_test = _synth(rng, 2000)
        cal = calibrate_escalation_threshold(conf_cal, correct_cal, alpha)
        accepted = conf_test >= cal.tau
        leaked_draws.append(float(np.mean(accepted & ~correct_test)))
    mean_leaked = float(np.mean(leaked_draws))
    assert mean_leaked <= alpha + 0.01


def test_weighted_shifts_threshold():
    """Weighting toward low-quality rows should not *loosen* the leaked-error control."""
    rng = np.random.default_rng(7)
    conf, correct = _synth(rng, 2000)
    # Upweight the wrong-but-confident rows: threshold should rise (escalate more).
    weights = np.where(~correct & (conf > 0.7), 5.0, 1.0)
    cal_w = calibrate_escalation_threshold(conf, correct, 0.1, weights=weights)
    cal_u = calibrate_escalation_threshold(conf, correct, 0.1)
    assert cal_w.tau >= cal_u.tau


def test_density_ratio_weights():
    p_test = np.array([0.5, 0.9, 0.1])
    w = density_ratio_weights(np.zeros(3), p_test)
    # w = p/(1-p): 1.0, 9.0, 0.111...
    assert np.allclose(w, [1.0, 9.0, 1 / 9], atol=1e-6)
