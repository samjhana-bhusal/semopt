"""Tests for the cascade dispatcher and calibration (FR-6, FR-7 integration)."""

from __future__ import annotations

import math

from semopt.cascade.cascade import Cascade, Tier, calibrate_cascade
from semopt.models.base import Model, ModelResponse
from semopt.models.mock import MockModel


class ScriptedModel(Model):
    """Returns a fixed (value, confidence) per prompt, with logprobs, for exact control."""

    def __init__(self, model_id: str, table: dict[str, tuple[str, float]], default=("no", 0.5)):
        self.model_id = model_id
        self.supports_logprobs = True
        self._table = table
        self._default = default

    def predict(self, prompt, *, examples=None, temperature=0.0, max_tokens=256, seed=None):
        value, conf = self._table.get(prompt, self._default)
        other = "no" if value == "yes" else "yes"
        logprobs = {value: math.log(conf), other: math.log(1 - conf)}
        return ModelResponse(
            value=value, tokens_in=10, tokens_out=1, wall_ms=0.0,
            logprobs=logprobs, model_id=self.model_id,
        )


def test_single_tier_accepts_all():
    tier = Tier(MockModel("mock-cheap", rule=lambda p: "yes"))
    casc = Cascade([tier])
    res = casc.run_row("anything")
    assert res.final_tier == 0
    assert res.value in ("yes", "no")


def test_escalation_when_low_confidence():
    # Cheap tier: low confidence on 'hard' → escalate. High on 'easy' → accept.
    cheap = ScriptedModel("mock-cheap", {"easy": ("yes", 0.95), "hard": ("no", 0.55)})
    expensive = ScriptedModel("gpt-4o", {"hard": ("yes", 0.99)})
    tiers = [Tier(cheap, tau=0.8), Tier(expensive)]  # final tier tau=-inf
    casc = Cascade(tiers)

    easy = casc.run_row("easy")
    assert easy.final_tier == 0  # accepted cheaply
    assert easy.value == "yes"

    hard = casc.run_row("hard")
    assert hard.final_tier == 1  # escalated to expensive
    assert hard.value == "yes"
    assert len(hard.history) == 2
    assert hard.history[0].accepted is False
    assert hard.history[1].accepted is True


def test_cost_accumulates_across_tiers():
    cheap = ScriptedModel("mock-cheap", {"hard": ("no", 0.5)})
    expensive = ScriptedModel("gpt-4o", {"hard": ("yes", 0.99)})
    casc = Cascade([Tier(cheap, tau=0.9), Tier(expensive)])
    res = casc.run_row("hard")
    # gpt-4o priced; mock-cheap free → total cost is the expensive tier's call cost.
    assert res.total_cost_usd > 0.0


def test_tier_distribution():
    cheap = ScriptedModel("mock-cheap", {"a": ("yes", 0.95), "b": ("no", 0.55)})
    expensive = ScriptedModel("gpt-4o", {})
    casc = Cascade([Tier(cheap, tau=0.8), Tier(expensive)])
    results = casc.run(["a", "a", "b"])
    dist = casc.tier_distribution(results)
    assert dist[0] == 2 / 3
    assert dist[1] == 1 / 3


def test_sem_filter_cascade_applies_calibrated_cascade():
    from semopt.table import SemanticTable

    table = SemanticTable.from_records([{"text": "a"}, {"text": "b"}, {"text": "c"}])
    expensive = ScriptedModel("gpt-4o", {}, default=("yes", 0.99))

    # Cheap confidently 'no' (conf 0.9 ≥ tau 0.5) → accepted at tier 0, all filtered out.
    cheap_no = ScriptedModel("mock-cheap", {}, default=("no", 0.9))
    out_none = table.sem_filter_cascade(
        "keep?", cascade=Cascade([Tier(cheap_no, tau=0.5), Tier(expensive)])
    )
    assert len(out_none) == 0
    assert out_none.columns == ["text"]

    # Cheap confidently 'yes' → all kept.
    cheap_yes = ScriptedModel("mock-cheap", {}, default=("yes", 0.9))
    out_all = table.sem_filter_cascade(
        "keep?", cascade=Cascade([Tier(cheap_yes, tau=0.5), Tier(expensive)])
    )
    assert len(out_all) == 3


def test_calibrate_cascade_sets_thresholds():
    # Cheap tier is confident+correct on 'easy*', unconfident+wrong on 'hard*'.
    table = {}
    labels = []
    prompts = []
    for i in range(50):
        p = f"easy{i}"
        table[p] = ("yes", 0.95)
        prompts.append(p)
        labels.append(True)  # correct
    for i in range(50):
        p = f"hard{i}"
        table[p] = ("yes", 0.55)  # confidently-ish wrong
        prompts.append(p)
        labels.append(False)  # actually should be 'no' → wrong

    cheap = ScriptedModel("mock-cheap", table)
    expensive = ScriptedModel("gpt-4o", {})
    tiers = [Tier(cheap), Tier(expensive)]

    def is_correct(value: str, label: object) -> bool:
        return (value == "yes") == bool(label)

    cals = calibrate_cascade(
        tiers, prompts, labels, alpha=0.1, is_correct=is_correct
    )
    assert len(cals) == 1  # only the non-final tier calibrated
    # Threshold must sit above the wrong rows' 0.55 confidence to escalate them.
    assert tiers[0].tau > 0.55
    assert cals[0].leaked_error_weight <= 0.1 + 1e-9
