"""Baselines B0–B5 for the cost–accuracy comparison (SRS §8).

All baselines are expressed as a :class:`~semopt.cascade.cascade.Cascade` over the same
cheap/expensive tiers, sharing one cost model so comparisons are fair (§8, last line):

* **B0 naive_strongest** — one tier, the expensive model (upper cost bound).
* **B1 naive_cheapest** — one tier, the cheap model (lower cost/accuracy bound).
* **B2 hand_tuned_cascade** — FrugalGPT-style: τ grid-searched on the cal set to hit the
  target accuracy (no coverage guarantee).
* **B3 fixed_threshold_cascade** — arbitrary τ = 0.8 (shows why a principled τ matters).
* **ours conformal_cascade** — τ from weighted-conformal calibration with the 2-tier
  budget (FR-7); the method under test.
* **B5 conformal_cascade_unweighted** — the weighted-conformal-disabled ablation (plain
  conformal). Identical to ours when there is no covariate shift, which is the point.

B4 (LOTUS) is an external repo and is documented-and-skipped here (SRS §8, Risk R4).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from semopt.cascade.cascade import Cascade, Tier
from semopt.cascade.confidence import logprob_confidence
from semopt.cascade.conformal import calibrate_escalation_threshold
from semopt.cost.cost_model import CostModel
from semopt.eval.metrics import binary_f1
from semopt.models.base import Model

IsCorrect = Callable[[str, object], bool]


def default_is_correct(value: str, label: object) -> bool:
    return (value == "yes") == bool(label)


@dataclass
class BaselineResult:
    name: str
    accuracy: float
    f1: float
    cost_usd: float
    escalate_frac: float
    tau: float | None = None


def _cheap_confidences(model: Model, prompts: Sequence[str]) -> np.ndarray:
    scores = []
    for p in prompts:
        conf = logprob_confidence(model.predict(p, max_tokens=4))
        scores.append(conf.score if conf is not None else 0.5)
    return np.array(scores)


def _cheap_correct(
    model: Model, prompts: Sequence[str], labels: Sequence[object], is_correct: IsCorrect
) -> np.ndarray:
    out = []
    for p, y in zip(prompts, labels, strict=True):
        conf = logprob_confidence(model.predict(p, max_tokens=4))
        val = conf.value if conf is not None else "no"
        out.append(is_correct(val, y))
    return np.array(out, dtype=bool)


def conformal_two_tier_tau(
    cheap: Model,
    expensive: Model,
    cal_prompts: Sequence[str],
    cal_labels: Sequence[object],
    *,
    alpha: float,
    is_correct: IsCorrect = default_is_correct,
    weights: np.ndarray | None = None,
    max_iter: int = 6,
) -> tuple[float, float]:
    """Calibrate the cheap tier's τ for a 2-tier conformal cascade (shared by experiments).

    Solves the fixed point α₀ = α − escalate_frac · err_strong so end-to-end error is
    controlled at α (see experiments/run_coverage.py for the derivation). Returns
    ``(tau, err_strong)``.
    """
    strong_correct = np.array(
        [is_correct(expensive.predict(p).value, y)
         for p, y in zip(cal_prompts, cal_labels, strict=True)]
    )
    err_strong = float(1.0 - strong_correct.mean())

    conf = _cheap_confidences(cheap, cal_prompts)
    correct = _cheap_correct(cheap, cal_prompts, cal_labels, is_correct)

    escalate = 1.0
    tau = 1.0
    for _ in range(max_iter):
        alpha0 = max(alpha - escalate * err_strong, 1e-3)
        cal = calibrate_escalation_threshold(conf, correct, alpha0, weights=weights)
        tau = cal.tau
        new_escalate = float(np.mean(conf < tau))
        if abs(new_escalate - escalate) < 1e-3:
            break
        escalate = new_escalate
    return tau, err_strong


def _evaluate(
    name: str,
    cascade: Cascade,
    prompts: Sequence[str],
    labels: Sequence[object],
    is_correct: IsCorrect,
    tau: float | None = None,
) -> BaselineResult:
    results = cascade.run(list(prompts))
    preds = [r.value for r in results]
    correct = [is_correct(p, y) for p, y in zip(preds, labels, strict=True)]
    acc = float(np.mean(correct)) if correct else 0.0
    y_true = [bool(y) for y in labels]
    y_pred = [p == "yes" for p in preds]
    f1 = binary_f1(y_true, y_pred).f1
    escalate = float(np.mean([r.final_tier > 0 for r in results])) if results else 0.0
    cost = float(sum(r.total_cost_usd for r in results))
    return BaselineResult(name, acc, f1, cost, escalate, tau)


def hand_tuned_tau(
    cheap: Model,
    cal_prompts: Sequence[str],
    cal_labels: Sequence[object],
    *,
    target_accuracy: float,
    is_correct: IsCorrect = default_is_correct,
    grid: Sequence[float] = tuple(np.round(np.arange(0.5, 1.0, 0.02), 3)),
) -> float:
    """FrugalGPT-style: smallest τ whose *cal-set accepted* accuracy ≥ target (B2).

    Mirrors picking a confidence cutoff by hand: among rows the cheap tier would accept at
    τ, what fraction are correct? Choose the least aggressive τ meeting the target.
    """
    conf = _cheap_confidences(cheap, cal_prompts)
    correct = _cheap_correct(cheap, cal_prompts, cal_labels, is_correct)
    best = 1.0
    for tau in sorted(grid):
        accepted = conf >= tau
        if accepted.sum() == 0:
            continue
        acc = correct[accepted].mean()
        if acc >= target_accuracy:
            best = float(tau)
            break
    return best


def run_all_baselines(
    cheap: Model,
    expensive: Model,
    cal_prompts: Sequence[str],
    cal_labels: Sequence[object],
    test_prompts: Sequence[str],
    test_labels: Sequence[object],
    *,
    alpha: float,
    cost_model: CostModel | None = None,
    is_correct: IsCorrect = default_is_correct,
) -> list[BaselineResult]:
    """Run B0–B3, our conformal cascade, and the B5 unweighted ablation."""
    cm = cost_model or CostModel.from_yaml()

    def cascade(tiers: list[Tier]) -> Cascade:
        return Cascade(tiers, cost_model=cm)

    results: list[BaselineResult] = []

    # B0 / B1: single-tier naive baselines.
    results.append(
        _evaluate(
            "B0_naive_strongest", cascade([Tier(expensive)]),
            test_prompts, test_labels, is_correct,
        )
    )
    results.append(
        _evaluate(
            "B1_naive_cheapest", cascade([Tier(cheap)]),
            test_prompts, test_labels, is_correct,
        )
    )

    # B3: fixed arbitrary threshold.
    results.append(
        _evaluate(
            "B3_fixed_threshold",
            cascade([Tier(cheap, tau=0.8), Tier(expensive)]),
            test_prompts, test_labels, is_correct, tau=0.8,
        )
    )

    # B2: hand-tuned threshold to hit target accuracy on cal.
    tau_ht = hand_tuned_tau(
        cheap, cal_prompts, cal_labels, target_accuracy=1 - alpha, is_correct=is_correct
    )
    results.append(
        _evaluate(
            "B2_hand_tuned",
            cascade([Tier(cheap, tau=tau_ht), Tier(expensive)]),
            test_prompts, test_labels, is_correct, tau=tau_ht,
        )
    )

    # ours: conformal cascade (weighted machinery; no shift here → weights=None).
    tau_conf, _ = conformal_two_tier_tau(
        cheap, expensive, cal_prompts, cal_labels, alpha=alpha, is_correct=is_correct
    )
    results.append(
        _evaluate(
            "ours_conformal",
            cascade([Tier(cheap, tau=tau_conf), Tier(expensive)]),
            test_prompts, test_labels, is_correct, tau=tau_conf,
        )
    )

    # B5: unweighted-conformal ablation (identical calibration path, weights=None) —
    # kept as an explicit line so the comparison table shows the ablation.
    results.append(
        _evaluate(
            "B5_conformal_unweighted",
            cascade([Tier(cheap, tau=tau_conf), Tier(expensive)]),
            test_prompts, test_labels, is_correct, tau=tau_conf,
        )
    )
    return results
