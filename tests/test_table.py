"""Tests for the SemanticTable abstraction (FR-1)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from semopt import SemanticTable
from semopt.table import ColumnProvenance


def _sample() -> SemanticTable:
    return SemanticTable.from_records(
        [
            {"id": 1, "text": "shipping was late"},
            {"id": 2, "text": "great product"},
            {"id": 3, "text": "box arrived crushed"},
        ]
    )


def test_from_records_and_access():
    t = _sample()
    assert len(t) == 3
    assert t.columns == ["id", "text"]
    assert t.shape == (3, 2)
    assert list(t["id"]) == [1, 2, 3]
    assert "text" in t
    assert "missing" not in t


def test_from_csv_json_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    assert SemanticTable.from_csv(csv).shape == (2, 2)

    pq = tmp_path / "t.parquet"
    df.to_parquet(pq)
    assert SemanticTable.from_parquet(pq).columns == ["a", "b"]

    # list-of-records JSON
    js = tmp_path / "recs.json"
    js.write_text(json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]))
    assert len(SemanticTable.from_json(js)) == 2

    # column-oriented JSON
    js2 = tmp_path / "cols.json"
    js2.write_text(json.dumps({"a": [1, 2], "b": ["x", "y"]}))
    assert SemanticTable.from_json(js2).shape == (2, 2)


def test_immutability():
    t = _sample()
    external = t.to_pandas()
    external.loc[0, "text"] = "MUTATED"
    # Mutating the returned frame must not affect the table.
    assert t["text"].iloc[0] == "shipping was late"


def test_source_provenance_default():
    t = _sample()
    assert t.provenance("id").operator == "source"
    assert t.total_cost_usd() == 0.0


def test_with_columns_records_provenance():
    t = _sample()
    prov = ColumnProvenance(operator="sem_map", model_id="m", cost_usd=0.5)
    t2 = t.with_columns({"sentiment": ["neg", "pos", "neg"]}, {"sentiment": prov})
    assert "sentiment" in t2
    assert t2.provenance("sentiment").cost_usd == 0.5
    assert t2.total_cost_usd() == 0.5
    # Original unchanged (immutability).
    assert "sentiment" not in t


def test_select_rows_folds_cost():
    t = _sample()
    prov = ColumnProvenance(operator="sem_filter", model_id="m", cost_usd=0.3)
    t2 = t.select_rows([True, False, True], prov)
    assert len(t2) == 2
    assert list(t2["id"]) == [1, 3]
    assert t2.total_cost_usd() == pytest.approx(0.3)
