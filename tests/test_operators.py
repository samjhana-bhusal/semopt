"""Tests for sem_filter and sem_map (FR-2.1, FR-2.2)."""

from __future__ import annotations

import pytest

from semopt import SemanticTable
from semopt.models.mock import KeywordFilterModel, MockModel
from semopt.operators.filter import parse_bool


def _reviews() -> SemanticTable:
    return SemanticTable.from_records(
        [
            {"id": 1, "text": "shipping was late and box was crushed"},
            {"id": 2, "text": "great product, love it"},
            {"id": 3, "text": "the shipping took forever"},
            {"id": 4, "text": "works as described"},
        ]
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("yes", True),
        ("Yes.", True),
        ("YES it does", True),
        ("no", False),
        ("No.", False),
        ("", False),
        ("maybe not", False),
        ("true", True),
        ("Absolutely, this is complaining", True),
    ],
)
def test_parse_bool(text, expected):
    assert parse_bool(text) is expected


def test_sem_filter_keeps_matching_rows():
    t = _reviews()
    # Keyword appears in rows 1 & 3 only; the question text avoids it so the mock's
    # decision is driven purely by the row content.
    model = KeywordFilterModel("mock-cheap", keyword="shipping")
    out = t.sem_filter("Is this review about a delivery problem?", model=model)
    assert sorted(out["id"]) == [1, 3]
    # Filter cost is folded into provenance total (mock-cheap is $0 but tokens counted).
    assert out.total_cost_usd() == 0.0


def test_sem_filter_cost_accumulates_on_priced_model():
    t = _reviews()
    model = KeywordFilterModel("mock-expensive", keyword="shipping")
    out = t.sem_filter("Does this review mention shipping?", model=model)
    assert out.total_cost_usd() > 0.0


def test_sem_map_adds_column():
    t = _reviews()
    model = MockModel("mock-cheap", rule=lambda p: "positive")
    out = t.sem_map("Sentiment?", model=model, output_column="sentiment")
    assert "sentiment" in out
    assert list(out["sentiment"]) == ["positive"] * 4
    assert out.provenance("sentiment").operator == "sem_map"
    assert len(out) == 4


def test_sem_filter_invalid_target_accuracy():
    t = _reviews()
    model = KeywordFilterModel("mock-cheap", keyword="x")
    with pytest.raises(ValueError):
        t.sem_filter("q", model=model, target_accuracy=1.5)
