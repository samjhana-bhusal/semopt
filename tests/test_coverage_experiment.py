"""Smoke + property test for the coverage experiment (SC2 / Milestone M1)."""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.synthetic import make_workload
from experiments.run_coverage import run_one

from semopt.cost.cost_model import CostModel


def test_synthetic_workload_deterministic():
    a = make_workload(200, seed=1)
    b = make_workload(200, seed=1)
    assert a.prompts == b.prompts
    assert a.labels == b.labels
    # Cheap tier is informative: confident rows are more often correct than unconfident.
    from semopt.cascade.confidence import logprob_confidence

    confs, correct = [], []
    for p, y in zip(a.prompts, a.labels, strict=True):
        c = logprob_confidence(a.cheap.predict(p))
        confs.append(c.score)
        correct.append((c.value == "yes") == y)
    confs = np.array(confs)
    correct = np.array(correct)
    hi = correct[confs >= np.median(confs)].mean()
    lo = correct[confs < np.median(confs)].mean()
    assert hi > lo  # higher confidence ⇒ higher accuracy


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_coverage_within_tolerance(alpha):
    cost_model = CostModel.from_yaml()
    r = run_one(alpha, n=1500, cost_model=cost_model)
    # SC2: empirical coverage within ±3% of nominal 1−α (small extra slack for n=1500).
    assert abs(r["coverage_gap"]) <= 0.04
    # Coverage should not fall materially below nominal (the guarantee direction).
    assert r["empirical_coverage"] >= (1 - alpha) - 0.03
    # Cost must be a genuine fraction of naive-strongest (cascade actually saves).
    assert 0.0 < r["cost_fraction"] < 1.0


def test_cost_savings_at_alpha_010():
    """SC1 flavor: at α=0.10 the cascade should cost well under naive-strongest."""
    r = run_one(0.10, n=1500, cost_model=CostModel.from_yaml())
    assert r["cost_fraction"] <= 0.30
