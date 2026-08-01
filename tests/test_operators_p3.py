"""Tests for the P3 operators: sem_extract, sem_join, sem_rank (FR-2.3-2.5)."""

from __future__ import annotations

import pytest

from semopt.models.base import Model, ModelResponse
from semopt.models.mock import MockModel
from semopt.operators.extract import parse_fields
from semopt.operators.rank import parse_score
from semopt.table import SemanticTable


class ProgrammableModel(Model):
    """Returns the response produced by ``fn(prompt)`` — for scripting per-prompt output."""

    def __init__(self, model_id: str, fn):  # type: ignore[no-untyped-def]
        self.model_id = model_id
        self.supports_logprobs = False
        self._fn = fn

    def predict(self, prompt, *, examples=None, temperature=0.0, max_tokens=256, seed=None):
        return ModelResponse(
            value=self._fn(prompt), tokens_in=10, tokens_out=5, wall_ms=0.0,
            model_id=self.model_id,
        )


# ----------------------------------------------------------------- sem_extract
def test_parse_fields_json_block():
    text = '```json\n{"type": "NDA", "risk": "high"}\n```'
    assert parse_fields(text, ["type", "risk"]) == {"type": "NDA", "risk": "high"}


def test_parse_fields_bare_object_and_missing():
    text = '{"type": "MSA"}'
    assert parse_fields(text, ["type", "risk"]) == {"type": "MSA", "risk": ""}


def test_parse_fields_line_fallback():
    text = "type: lease\nrisk: low"
    assert parse_fields(text, ["type", "risk"]) == {"type": "lease", "risk": "low"}


def test_sem_extract_produces_columns():
    table = SemanticTable.from_records(
        [{"text": "a contract"}, {"text": "another"}]
    )
    model = ProgrammableModel("gpt-4o", lambda p: '{"clause": "indemnity", "risk": "high"}')
    out = table.sem_extract("extract", fields=["clause", "risk"], model=model)
    assert "clause" in out.columns and "risk" in out.columns
    assert list(out["clause"]) == ["indemnity", "indemnity"]
    assert out.provenance("clause").operator == "sem_extract"
    # Cost is split across produced columns (not double-counted).
    assert out.provenance("clause").cost_usd > 0


def test_sem_extract_requires_fields():
    table = SemanticTable.from_records([{"text": "x"}])
    with pytest.raises(ValueError):
        table.sem_extract("e", fields=[], model=MockModel("m", rule=lambda p: "no"))


# -------------------------------------------------------------------- sem_join
def test_sem_join_nested_loop():
    left = SemanticTable.from_records([{"name": "Bob"}, {"name": "Sue"}])
    right = SemanticTable.from_records([{"person": "Bob"}, {"person": "Sue"}])

    # Match when the LEFT name appears in the RIGHT block of the prompt.
    def fn(prompt: str) -> str:
        left_part, _, right_part = prompt.partition("--- RIGHT ---")
        lname = left_part.split("name:")[1].split()[0]
        return "yes" if f"person: {lname}" in right_part else "no"

    model = ProgrammableModel("gpt-4o", fn)
    out = left.sem_join(right, "same person?", model=model)
    assert len(out) == 2  # Bob-Bob, Sue-Sue
    assert set(out.columns) == {"name", "person"}


def test_sem_join_blocked_reduces_pairs():
    left = SemanticTable.from_records([{"id": 1, "v": "a"}, {"id": 2, "v": "b"}])
    right = SemanticTable.from_records([{"id": 1, "w": "x"}, {"id": 2, "w": "y"}])
    calls = {"n": 0}

    def fn(prompt: str) -> str:
        calls["n"] += 1
        return "yes"

    model = ProgrammableModel("gpt-4o", fn)
    out = left.sem_join(
        right, "match?", model=model, how="blocked", left_block="id", right_block="id"
    )
    # Only same-id pairs evaluated: 2 pairs, not 4.
    assert calls["n"] == 2
    assert len(out) == 2
    assert "id" in out.columns and "id_right" in out.columns


def test_sem_join_blocked_requires_keys():
    left = SemanticTable.from_records([{"id": 1}])
    with pytest.raises(ValueError):
        left.sem_join(left, "m", model=MockModel("m", rule=lambda p: "no"), how="blocked")


# -------------------------------------------------------------------- sem_rank
def test_parse_score_clamps():
    assert parse_score("relevance: 7") == 7.0
    assert parse_score("12") == 10.0
    assert parse_score("-3") == 0.0
    assert parse_score("no number") == 0.0


def test_sem_rank_pointwise_topk():
    rows = [{"text": f"doc {i}", "want": i} for i in range(5)]
    table = SemanticTable.from_records(rows)
    # Score = the row's 'want' value, embedded in the rendered prompt.
    def fn(prompt: str) -> str:
        want = prompt.rsplit("want:", 1)[1].strip().split()[0]
        return want

    model = ProgrammableModel("gpt-4o", fn)
    out = table.sem_rank("most relevant?", k=2, model=model)
    assert len(out) == 2
    assert "rank_score" in out.columns
    # Highest 'want' (4, then 3) come first.
    assert list(out["want"]) == [4, 3]


def test_sem_rank_pairwise_copeland_handles_cycle():
    # Three items with an intransitive judge: A>B, B>C, C>A (a 3-cycle).
    table = SemanticTable.from_records([{"text": "A"}, {"text": "B"}, {"text": "C"}])

    def fn(prompt: str) -> str:
        a = prompt.split("--- A ---")[1].split("text:")[1].split()[0]
        b = prompt.split("--- B ---")[1].split("text:")[1].split()[0]
        beats = {("A", "B"), ("B", "C"), ("C", "A")}
        return "A" if (a, b) in beats else "B"

    model = ProgrammableModel("gpt-4o", fn)
    out = table.sem_rank("rank", k=3, method="pairwise", model=model)
    # Each wins exactly once → all tied at 1.0; Copeland leaves order stable, no crash.
    assert len(out) == 3
    assert set(out["rank_score"]) == {1.0}


def test_sem_rank_invalid_args():
    table = SemanticTable.from_records([{"text": "x"}])
    with pytest.raises(ValueError):
        table.sem_rank("q", k=0, model=MockModel("m", rule=lambda p: "no"))
    with pytest.raises(ValueError):
        table.sem_rank("q", k=1, method="bogus", model=MockModel("m", rule=lambda p: "no"))
