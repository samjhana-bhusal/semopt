"""Tests for eval metrics and baselines (SRS §8, §9)."""

from __future__ import annotations

import numpy as np
from benchmarks.synthetic import make_split_workload

from semopt.eval.baselines import hand_tuned_tau, run_all_baselines
from semopt.eval.metrics import (
    binary_f1,
    bootstrap_ci,
    exact_match,
    mean_span_f1,
    ndcg_at_k,
    span_f1,
)


# ----------------------------------------------------------------- metrics
def test_binary_f1_perfect_and_none():
    perfect = binary_f1([True, False, True], [True, False, True])
    assert perfect.f1 == 1.0 and perfect.precision == 1.0 and perfect.recall == 1.0
    none = binary_f1([True, True], [False, False])
    assert none.f1 == 0.0


def test_binary_f1_partial():
    # tp=1, fp=1, fn=1 → p=r=0.5, f1=0.5
    prf = binary_f1([True, True, False], [True, False, True])
    assert abs(prf.f1 - 0.5) < 1e-9


def test_exact_match_normalizes():
    assert exact_match(["Hello ", "b"], ["hello", "B"]) == 1.0
    assert exact_match(["a", "b"], ["a", "c"]) == 0.5


def test_span_f1():
    assert span_f1("high risk", "high risk") == 1.0
    assert span_f1("high risk clause", "risk") > 0.0
    assert span_f1("foo", "bar") == 0.0
    assert mean_span_f1(["a b", "c"], ["a b", "c"]) == 1.0


def test_ndcg_at_k():
    # Ideal ordering → NDCG 1.0; reversed → < 1.0.
    assert abs(ndcg_at_k([3, 2, 1], 3) - 1.0) < 1e-9
    assert ndcg_at_k([1, 2, 3], 3) < 1.0
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_bootstrap_ci_brackets_mean():
    vals = list(np.random.default_rng(0).normal(0.9, 0.05, 500))
    lo, hi = bootstrap_ci(vals, seed=1)
    assert lo < np.mean(vals) < hi


# --------------------------------------------------------------- baselines
def test_hand_tuned_tau_in_range():
    wl = make_split_workload(400, seed=1)
    tau = hand_tuned_tau(
        wl.cheap, wl.cal_prompts, wl.cal_labels, target_accuracy=0.9
    )
    assert 0.5 <= tau <= 1.0


def test_run_all_baselines_orders_make_sense():
    wl = make_split_workload(1000, seed=3)
    results = {
        r.name: r
        for r in run_all_baselines(
            wl.cheap, wl.expensive,
            wl.cal_prompts, wl.cal_labels,
            wl.test_prompts, wl.test_labels,
            alpha=0.10,
        )
    }
    strong = results["B0_naive_strongest"]
    cheap = results["B1_naive_cheapest"]
    ours = results["ours_conformal"]

    # Strong model is the most accurate and (per row) the most expensive.
    assert strong.accuracy >= ours.accuracy >= cheap.accuracy
    assert cheap.cost_usd < ours.cost_usd < strong.cost_usd
    # Our cascade escalates only some rows.
    assert 0.0 < ours.escalate_frac < 1.0
    # B0/B1 never escalate (single tier).
    assert strong.escalate_frac == 0.0 and cheap.escalate_frac == 0.0


def test_conformal_meets_target_cheaper_than_naive():
    wl = make_split_workload(1500, seed=5)
    results = {
        r.name: r
        for r in run_all_baselines(
            wl.cheap, wl.expensive,
            wl.cal_prompts, wl.cal_labels,
            wl.test_prompts, wl.test_labels,
            alpha=0.10,
        )
    }
    ours = results["ours_conformal"]
    strong = results["B0_naive_strongest"]
    # Meets the 1-α accuracy target within tolerance, at a fraction of naive cost (SC1).
    assert ours.accuracy >= 0.90 - 0.03
    assert ours.cost_usd <= 0.60 * strong.cost_usd


# --------------------------------------------------------------- loaders
def test_benchmark_loaders_are_documented_stubs():
    import pytest
    from benchmarks.loader import LOADERS

    assert set(LOADERS) == {"product_reviews", "arxiv_abstracts", "contract_review"}
    for name, fn in LOADERS.items():
        with pytest.raises(NotImplementedError) as exc:
            fn()
        # The error must tell the user how to wire it up (not just fail silently).
        assert "To enable it" in str(exc.value)
        assert name in str(exc.value)
