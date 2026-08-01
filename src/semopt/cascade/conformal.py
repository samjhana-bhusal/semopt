"""Conformal calibration of cascade escalation thresholds (FR-7) — the paper's core.

Adapts the weighted-quantile machinery from the applicant's Drift-Lab project
(``proj2/src/drift_lab/analysis/conformal.py``, Tibshirani et al. 2019). That code builds
*prediction sets*; here we reuse the same weighted-empirical-quantile idea to derive a
scalar **escalation threshold** τ per tier.

Guarantee (made precise — the SRS §7.2 sketch left the quantile direction ambiguous, so
this is the sharpened version; see NOTES.md):

    At a tier, we ACCEPT the cheap answer when ``confidence >= τ`` and otherwise ESCALATE.
    An accepted answer is a *leaked error* if the model was actually wrong. We choose the
    smallest τ (i.e. accept as many rows as cheaply as possible) such that, on
    exchangeable test data,

        P(model wrong  AND  confidence >= τ)  <=  α .

    Under covariate shift between calibration and deployment, calibration points carry
    density-ratio weights w(x) = p_test(x) / p_train(x); the bound then holds for the
    shifted test distribution (Tibshirani et al. 2019). This controls the error *added*
    by accepting cheaply; escalated rows are handled by a stronger tier, and end-to-end
    coverage is validated empirically (FR-7.3, SC2).

The threshold is a weighted quantile of the confidences of the *incorrect* calibration
points, with a finite-sample correction that treats the unknown test point conservatively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

# τ is nudged just above the confidence of the first non-leaked error so that point is
# escalated under the ``confidence >= τ`` rule (avoids a boundary tie).
_TIE_EPS = 1e-9


@dataclass(frozen=True)
class TierCalibration:
    """Result of calibrating one tier's escalation threshold."""

    tau: float
    alpha: float
    n_cal: int
    n_errors: int
    accept_rate: float  # fraction of calibration points accepted (not escalated)
    leaked_error_weight: float  # weighted P(wrong AND accept) achieved on calibration


def calibrate_escalation_threshold(
    confidence: npt.ArrayLike,
    correct: npt.ArrayLike,
    alpha: float,
    weights: npt.ArrayLike | None = None,
) -> TierCalibration:
    """Derive a tier's escalation threshold τ controlling leaked error at level ``alpha``.

    Parameters
    ----------
    confidence : shape (n,), each in [0, 1] — the tier's per-row confidence (FR-5).
    correct    : shape (n,), bool — whether the tier's answer was correct on that row.
    alpha      : target leaked-error rate, ``P(wrong AND confidence >= τ) <= alpha``.
    weights    : shape (n,), optional density-ratio weights for covariate shift. Defaults
                 to unweighted (standard exchangeable conformal).

    Returns
    -------
    TierCalibration with ``tau`` the escalation threshold: accept when
    ``confidence >= tau``, else escalate.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    conf = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    n = conf.shape[0]
    if n == 0:
        raise ValueError("empty calibration set")
    if ok.shape[0] != n:
        raise ValueError("confidence and correct must have the same length")

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.clip(np.asarray(weights, dtype=np.float64), 1e-8, None)
        if w.shape[0] != n:
            raise ValueError("weights must match confidence length")

    total_weight = float(w.sum())
    wrong = ~ok
    n_errors = int(wrong.sum())

    # No errors observed: accept everything (never escalate). τ = 0 accepts all conf >= 0.
    if n_errors == 0:
        return TierCalibration(
            tau=0.0,
            alpha=alpha,
            n_cal=n,
            n_errors=0,
            accept_rate=1.0,
            leaked_error_weight=0.0,
        )

    wrong_conf = conf[wrong]
    wrong_w = w[wrong]

    # Sort the incorrect points by confidence, highest first: these are the errors most
    # likely to be (wrongly) accepted, so they consume the leak budget first.
    order = np.argsort(-wrong_conf, kind="stable")
    wc_sorted = wrong_conf[order]
    ww_sorted = wrong_w[order]
    cum_leak = np.cumsum(ww_sorted)

    # Finite-sample correction: an unknown exchangeable test point could itself be a
    # high-confidence error. Reserve one test-point's worth of weight (the max weight is
    # the conservative proxy) so the bound holds for the (n+1)-point exchangeable set.
    w_test = float(w.max())
    budget = alpha * (total_weight + w_test) - w_test
    budget = max(budget, 0.0)

    # k = how many top-confidence errors we may accept (leak) within budget.
    k = int(np.searchsorted(cum_leak, budget, side="right"))

    # k >= n_errors: even leaking every error stays within budget → accept all (τ=0).
    # Otherwise the k-th error (0-indexed) must escalate, so put τ just above its conf.
    tau = 0.0 if k >= n_errors else float(wc_sorted[k]) + _TIE_EPS

    accepted = conf >= tau
    accept_rate = float(w[accepted].sum() / total_weight)
    leaked_weight = float(w[accepted & wrong].sum() / total_weight)

    log.info(
        "tier calibrated | alpha=%.3f tau=%.4f n=%d errors=%d accept=%.3f leaked=%.4f",
        alpha,
        tau,
        n,
        n_errors,
        accept_rate,
        leaked_weight,
    )
    return TierCalibration(
        tau=tau,
        alpha=alpha,
        n_cal=n,
        n_errors=n_errors,
        accept_rate=accept_rate,
        leaked_error_weight=leaked_weight,
    )


def density_ratio_weights(
    cal_scores: FloatArray,
    discriminator_prob_test: FloatArray,
    clip: tuple[float, float] = (1e-3, 1e3),
) -> FloatArray:
    """Convert a train-vs-deploy discriminator's P(test | x) into density-ratio weights.

    ``w(x) = p_test(x) / p_train(x) = P(test | x) / (1 - P(test | x))`` for a balanced
    discriminator (FR-7.2 step 5). Clipped for numerical stability. ``cal_scores`` is
    accepted for API symmetry with proj2 and is currently unused beyond a length check.
    """
    p = np.clip(np.asarray(discriminator_prob_test, dtype=np.float64), 1e-6, 1 - 1e-6)
    if cal_scores is not None and len(cal_scores) != len(p):
        raise ValueError("cal_scores and discriminator_prob_test length mismatch")
    w = p / (1.0 - p)
    return np.clip(w, clip[0], clip[1])
