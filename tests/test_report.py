"""Tests for the trace (FR-10.2) and dashboard (FR-10.3) reports."""

from __future__ import annotations

import json

from semopt.cascade.cascade import Cascade, Tier
from semopt.models.mock import LookupModel
from semopt.report.dashboard import compute_stats, render_dashboard, write_dashboard
from semopt.report.trace import result_to_dict, write_trace


def _cascade_and_results():
    cheap = LookupModel(
        "mlx-llama-3.2-1b",
        {"easy": ("yes", 0.95), "hard": ("no", 0.55)},
        default=("no", 0.9),
    )
    expensive = LookupModel("gpt-4o", {"hard": ("yes", 0.99)}, default=("yes", 0.95))
    casc = Cascade([Tier(cheap, tau=0.8), Tier(expensive)])
    results = casc.run(["easy", "hard", "easy"])
    return casc, results


def test_result_to_dict_shape():
    _, results = _cascade_and_results()
    d = result_to_dict(0, results[0])
    assert d["row"] == 0
    assert d["final_tier"] == 0  # 'easy' accepted at tier 0
    assert isinstance(d["escalation_history"], list)
    assert d["escalation_history"][0]["model_id"] == "mlx-llama-3.2-1b"


def test_write_trace_is_valid_jsonl(tmp_path):
    _, results = _cascade_and_results()
    path = tmp_path / "trace.jsonl"
    n = write_trace(results, path)
    assert n == 3
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    # Every line parses; the 'hard' row escalated to tier 1.
    records = [json.loads(ln) for ln in lines]
    assert records[1]["final_tier"] == 1
    assert records[1]["escalation_history"][0]["accepted"] is False


def test_compute_stats_tier_and_cost():
    _, results = _cascade_and_results()
    stats = compute_stats(results, correct=[True, True, False], alpha=0.10)
    # 2 of 3 rows served by tier 0, 1 escalated to tier 1.
    assert abs(stats.tier_fractions[0] - 2 / 3) < 1e-9
    assert abs(stats.tier_fractions[1] - 1 / 3) < 1e-9
    assert stats.empirical_coverage == 2 / 3
    assert stats.alpha == 0.10
    assert stats.total_cost_usd >= 0.0


def test_render_dashboard_is_self_contained_html():
    _, results = _cascade_and_results()
    stats = compute_stats(results, correct=[True, True, True], alpha=0.10)
    doc = render_dashboard(stats, title="Test run")
    assert doc.startswith("<!doctype html>")
    assert "Tier distribution" in doc
    assert "Coverage vs. target" in doc  # coverage block present when correct given
    assert "http://" not in doc and "https://" not in doc  # no external assets


def test_write_dashboard_creates_file(tmp_path):
    _, results = _cascade_and_results()
    path = tmp_path / "dash.html"
    stats = write_dashboard(results, path, title="X", correct=[True, False, True], alpha=0.2)
    assert path.exists()
    assert stats.n_rows == 3


def test_dashboard_without_ground_truth_omits_coverage():
    _, results = _cascade_and_results()
    stats = compute_stats(results)
    assert stats.empirical_coverage is None
    doc = render_dashboard(stats, title="No labels")
    assert "Coverage vs. target" not in doc
