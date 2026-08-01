"""Tests for the cost model (FR-4)."""

from __future__ import annotations

import pytest

from semopt.cost.cost_model import CostModel


def test_loads_default_yaml():
    cm = CostModel.from_yaml()
    assert "gpt-4o" in cm
    assert "mock-cheap" in cm


def test_call_cost():
    cm = CostModel.from_yaml()
    # gpt-4o: $2.50/1M in, $10/1M out
    cost = cm.call_cost("gpt-4o", 1_000_000, 1_000_000)
    assert cost == pytest.approx(12.50)


def test_local_models_are_free():
    cm = CostModel.from_yaml()
    assert cm.call_cost("mlx-llama-3.2-1b", 10_000, 500) == 0.0


def test_estimate_operator_cost_scales_with_rows():
    cm = CostModel.from_yaml()
    d1, ms1 = cm.estimate_operator_cost("gpt-4o", 100)
    d2, ms2 = cm.estimate_operator_cost("gpt-4o", 200)
    assert d2 == pytest.approx(2 * d1)
    assert ms2 == pytest.approx(2 * ms1)


def test_unknown_model_raises():
    cm = CostModel.from_yaml()
    with pytest.raises(KeyError):
        cm.get("no-such-model")
