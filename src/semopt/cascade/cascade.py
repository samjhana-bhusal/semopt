"""Confidence-cascade dispatcher (FR-6).

A cascade is an ordered list of :class:`Tier` s (cheap → mid → expensive). For each row
it runs tier 1, scores confidence (FR-5), and accepts if ``confidence >= τ``; otherwise
it escalates to the next tier. The final tier has ``τ = -inf`` and always accepts. Every
row's decision is recorded (FR-6.4): final answer, final tier, per-tier escalation
history, and total cost.

Thresholds come from :func:`calibrate_cascade`, which runs the cheap tiers on a labeled
calibration set and calls the conformal calibrator (FR-7). They are never hand-tuned
(FR-6.3).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from semopt.cascade.confidence import Confidence, score_response
from semopt.cascade.conformal import TierCalibration, calibrate_escalation_threshold
from semopt.cost.cost_model import CostModel
from semopt.models.base import Model

# Predicate deciding whether a tier's answer matches ground truth during calibration.
IsCorrect = Callable[[str, object], bool]


@dataclass
class Tier:
    """One rung of the cascade: a model, whether its op is classification, and τ."""

    model: Model
    classification: bool = True
    tau: float = float("-inf")  # accept-all until calibrated; final tier stays -inf


@dataclass(frozen=True)
class TierVisit:
    tier_index: int
    model_id: str
    value: str
    score: float
    method: str
    accepted: bool
    cost_usd: float


@dataclass(frozen=True)
class CascadeResult:
    """Per-row outcome of running the cascade (FR-6.4)."""

    value: str
    final_tier: int
    confidence: float
    total_cost_usd: float
    tokens_in: int
    tokens_out: int
    history: tuple[TierVisit, ...] = field(default_factory=tuple)


class Cascade:
    """Runs a fixed, calibrated tier list per row."""

    def __init__(
        self,
        tiers: Sequence[Tier],
        *,
        cost_model: CostModel | None = None,
        k_self_consistency: int = 5,
    ) -> None:
        if not tiers:
            raise ValueError("cascade needs at least one tier")
        self.tiers = list(tiers)
        self.cost_model = cost_model or CostModel.from_yaml()
        self.k = k_self_consistency

    def _tier_cost(self, model_id: str, tokens_in: int, tokens_out: int) -> float:
        if model_id in self.cost_model:
            return self.cost_model.call_cost(model_id, tokens_in, tokens_out)
        return 0.0

    def run_row(self, prompt: str) -> CascadeResult:
        history: list[TierVisit] = []
        total_cost = 0.0
        total_in = 0
        total_out = 0
        last = len(self.tiers) - 1

        for idx, tier in enumerate(self.tiers):
            resp = tier.model.predict(prompt, max_tokens=(4 if tier.classification else 256))
            conf: Confidence = score_response(
                tier.model,
                prompt,
                resp,
                classification=tier.classification,
                k=self.k,
            )
            cost = self._tier_cost(tier.model.model_id, resp.tokens_in, resp.tokens_out)
            total_cost += cost
            total_in += resp.tokens_in
            total_out += resp.tokens_out
            accepted = idx == last or conf.score >= tier.tau
            history.append(
                TierVisit(
                    tier_index=idx,
                    model_id=tier.model.model_id,
                    value=conf.value,
                    score=conf.score,
                    method=conf.method,
                    accepted=accepted,
                    cost_usd=cost,
                )
            )
            if accepted:
                return CascadeResult(
                    value=conf.value,
                    final_tier=idx,
                    confidence=conf.score,
                    total_cost_usd=total_cost,
                    tokens_in=total_in,
                    tokens_out=total_out,
                    history=tuple(history),
                )

        raise AssertionError("unreachable: final tier always accepts")

    def run(self, prompts: Sequence[str]) -> list[CascadeResult]:
        return [self.run_row(p) for p in prompts]

    def tier_distribution(self, results: Sequence[CascadeResult]) -> dict[int, float]:
        """Fraction of rows served by each tier (Eval Metric 5 / plot P3)."""
        counts = np.zeros(len(self.tiers))
        for r in results:
            counts[r.final_tier] += 1
        total = max(len(results), 1)
        return {i: float(counts[i] / total) for i in range(len(self.tiers))}


def calibrate_cascade(
    tiers: Sequence[Tier],
    cal_prompts: Sequence[str],
    cal_labels: Sequence[object],
    *,
    alpha: float,
    is_correct: IsCorrect,
    weights: Sequence[float] | None = None,
    k_self_consistency: int = 5,
) -> list[TierCalibration]:
    """Calibrate every non-final tier's τ on a labeled calibration set (FR-7.2).

    Runs tier 1 on all calibration rows, computes ``(confidence, correct)``, and derives
    τ₁ via :func:`calibrate_escalation_threshold`. Rows accepted at tier 1 leave the
    calibration flow; the *escalated* rows carry forward to calibrate tier 2, and so on
    (FR-7.2 step 4). The final tier is left at ``τ = -inf`` (accept all). Mutates each
    non-final :class:`Tier` in place with its calibrated ``tau`` and returns the
    per-tier calibration records.

    ``weights`` are per-row density-ratio weights for covariate shift (FR-7.2 step 5);
    they are carried through escalation so downstream tiers stay correctly weighted.
    """
    n = len(cal_prompts)
    if len(cal_labels) != n:
        raise ValueError("cal_prompts and cal_labels length mismatch")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)

    active = np.arange(n)  # indices still flowing through the cascade
    calibrations: list[TierCalibration] = []

    for idx, tier in enumerate(tiers):
        if idx == len(tiers) - 1:
            break  # final tier accepts everything; nothing to calibrate
        if active.size == 0:
            break

        confidences = np.empty(active.size)
        correct = np.empty(active.size, dtype=bool)
        for j, row_np in enumerate(active):
            row = int(row_np)
            prompt = cal_prompts[row]
            resp = tier.model.predict(
                prompt, max_tokens=(4 if tier.classification else 256)
            )
            conf = score_response(
                tier.model, prompt, resp, classification=tier.classification,
                k=k_self_consistency,
            )
            confidences[j] = conf.score
            correct[j] = is_correct(conf.value, cal_labels[row])

        cal = calibrate_escalation_threshold(
            confidences, correct, alpha, weights=w[active]
        )
        tier.tau = cal.tau
        calibrations.append(cal)

        # Rows accepted here exit; the rest escalate to the next tier's calibration.
        escalated_mask = confidences < cal.tau
        active = active[escalated_mask]

    return calibrations
