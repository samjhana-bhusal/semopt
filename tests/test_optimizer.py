"""Tests for the logical plan, optimizer passes, and physical execution (FR-8, FR-10.1)."""

from __future__ import annotations

from semopt.cascade.cascade import Cascade, Tier
from semopt.cost.cost_model import CostModel
from semopt.models.mock import KeywordFilterModel, MockModel
from semopt.planner.logical import Filter, MapOp, Project
from semopt.planner.optimizer import Optimizer, OptimizerConfig, plan_cost
from semopt.planner.selectivity import SelectivityEstimator
from semopt.table import SemanticTable


def _table(keyword_freq: dict[str, int], n_each: int = 1) -> SemanticTable:
    """Build a table where each keyword appears in the requested number of rows."""
    rows = []
    for kw, count in keyword_freq.items():
        for _ in range(count * n_each):
            rows.append({"text": f"this row is about {kw}", "extra": "x"})
    return SemanticTable.from_records(rows)


# ------------------------------------------------------------------ builder
def test_query_builder_is_immutable():
    t = _table({"a": 2})
    q0 = t.query()
    q1 = q0.filter("about a?", model=KeywordFilterModel("m", "a"), column="text")
    assert len(q0.ops) == 0  # original unchanged
    assert len(q1.ops) == 1
    assert isinstance(q1.ops[0], Filter)


def test_filter_requires_model_or_cascade():
    import pytest

    with pytest.raises(ValueError):
        _table({"a": 1}).query().filter("x")


# ------------------------------------------------------------------ pushdown
def test_pushdown_moves_specific_filter_before_project():
    t = _table({"a": 3})
    q = (
        t.query()
        .project(["text"])
        .filter("about a?", model=KeywordFilterModel("m", "a"), column="text")
    )
    cfg = OptimizerConfig(reorder=False, memoize=False, cascade=False)
    opt = Optimizer(cfg).optimize(q)
    assert isinstance(opt.ops[0], Filter)  # filter pushed to front
    assert isinstance(opt.ops[1], Project)


def test_pushdown_blocked_when_readsall_filter_below_project():
    t = _table({"a": 3})
    # column=None → reads_all; project drops 'extra' → cannot move filter before it.
    q = t.query().project(["text"]).filter("about a?", model=KeywordFilterModel("m", "a"))
    cfg = OptimizerConfig(reorder=False, memoize=False, cascade=False)
    opt = Optimizer(cfg).optimize(q)
    assert isinstance(opt.ops[0], Project)  # unchanged: filter stays after project
    assert isinstance(opt.ops[1], Filter)


def test_pushdown_blocked_when_filter_reads_map_output():
    t = _table({"a": 3})
    m = MockModel("m", rule=lambda p: "label")
    q = (
        t.query()
        .map("label it", output_column="lab", model=m, column="text")
        .filter("is lab positive?", model=KeywordFilterModel("m", "x"), column="lab")
    )
    cfg = OptimizerConfig(reorder=False, memoize=False, cascade=False)
    opt = Optimizer(cfg).optimize(q)
    assert isinstance(opt.ops[0], MapOp)  # filter depends on map output → no move
    assert isinstance(opt.ops[1], Filter)


# ------------------------------------------------------------------ reorder
def test_reorder_reduces_estimated_cost():
    # 'alpha' in 90 rows (unselective), 'beta' in 10 rows (very selective).
    t = _table({"alpha": 90, "beta": 10})
    cost_model = CostModel.from_yaml()
    est = SelectivityEstimator(t, sample_size=200)  # sample all → exact selectivity

    expensive_unselective = KeywordFilterModel("gpt-4o", "alpha")
    cheap_selective = KeywordFilterModel("gpt-4o-mini", "beta")

    # Bad order: expensive+unselective filter first.
    bad = (
        t.query()
        .filter("alpha?", model=expensive_unselective, column="text")
        .filter("beta?", model=cheap_selective, column="text")
    )
    cfg = OptimizerConfig(pushdown=False, memoize=False, cascade=False, sample_size=200)
    good = Optimizer(cfg).optimize(bad)

    cost_bad = plan_cost(bad, cost_model, est).total_usd
    cost_good = plan_cost(good, cost_model, est).total_usd
    assert cost_good < cost_bad
    # The selective, cheap filter should now run first.
    assert good.ops[0].model.model_id == "gpt-4o-mini"


def test_reorder_unselective_filter_runs_last():
    t = _table({"alpha": 100, "beta": 20})  # 'alpha' in all → selectivity ~1.0
    cfg = OptimizerConfig(pushdown=False, memoize=False, cascade=False, sample_size=200)
    q = (
        t.query()
        .filter("alpha?", model=KeywordFilterModel("gpt-4o-mini", "alpha"), column="text")
        .filter("beta?", model=KeywordFilterModel("gpt-4o-mini", "beta"), column="text")
    )
    opt = Optimizer(cfg).optimize(q)
    # 'alpha' keeps everything (rank inf) → must be ordered last.
    assert opt.ops[-1].prompt == "alpha?"


# ------------------------------------------------------------------ memoize
def test_memoization_marks_duplicated_inputs():
    t = _table({"a": 1}, n_each=5)  # 5 identical rows → duplicates
    q = t.query().filter("about a?", model=KeywordFilterModel("m", "a"), column="text")
    cfg = OptimizerConfig(reorder=False, pushdown=False, cascade=False)
    opt = Optimizer(cfg).optimize(q)
    assert opt.ops[0].memoize is True


def test_no_memoization_without_duplicates():
    t = SemanticTable.from_records([{"text": "unique one"}, {"text": "unique two"}])
    q = t.query().filter("q?", model=KeywordFilterModel("m", "a"), column="text")
    cfg = OptimizerConfig(reorder=False, pushdown=False, cascade=False)
    opt = Optimizer(cfg).optimize(q)
    assert opt.ops[0].memoize is False


# ------------------------------------------------------------------ cascade
def test_cascade_insertion_via_factory():
    t = _table({"a": 3})
    model = KeywordFilterModel("mlx-llama-3.2-1b", "a")
    expensive = KeywordFilterModel("gpt-4o", "a")

    def factory(flt: Filter) -> Cascade:
        return Cascade([Tier(model, tau=0.8), Tier(expensive)])

    q = t.query().filter("about a?", model=model, column="text", use_cascade=True)
    cfg = OptimizerConfig(
        reorder=False, pushdown=False, memoize=False, cascade_factory=factory
    )
    opt = Optimizer(cfg).optimize(q)
    assert opt.ops[0].cascade is not None


def test_cascade_not_inserted_when_opted_out():
    t = _table({"a": 3})
    model = KeywordFilterModel("m", "a")

    def factory(flt: Filter) -> Cascade:  # pragma: no cover - must not be called
        raise AssertionError("factory should not run for use_cascade=False")

    q = t.query().filter("about a?", model=model, column="text", use_cascade=False)
    cfg = OptimizerConfig(reorder=False, pushdown=False, memoize=False, cascade_factory=factory)
    opt = Optimizer(cfg).optimize(q)
    assert opt.ops[0].cascade is None


# ------------------------------------------------------------------ execution
def test_optimized_and_unoptimized_agree():
    t = _table({"alpha": 5, "beta": 5, "gamma": 5}, n_each=2)
    q = (
        t.query()
        .filter("alpha?", model=KeywordFilterModel("m", "alpha"), column="text", use_cascade=False)
        .filter("beta?", model=KeywordFilterModel("m", "beta"), column="text", use_cascade=False)
    )
    # No row is about both alpha and beta → both plans yield 0 rows, but identically.
    naive = q.collect(optimize=False)
    opt = q.collect(optimize=True)
    assert len(naive) == len(opt)
    assert set(naive.to_pandas()["text"]) == set(opt.to_pandas()["text"])


def test_unique_fraction_reflects_duplicates():
    t = _table({"a": 1}, n_each=4)  # 4 identical rows
    est = SelectivityEstimator(t)
    op = Filter(reads=frozenset({"text"}), reads_all=False, is_llm=True)
    assert est.unique_fraction(op) == 0.25  # 1 distinct / 4 rows


def test_plan_cost_credits_memoization():
    t = _table({"a": 1}, n_each=10)  # 10 identical rows → 90% cache hits
    cost_model = CostModel.from_yaml()
    est = SelectivityEstimator(t, sample_size=200)
    m = KeywordFilterModel("gpt-4o", "a")

    q_plain = t.query().filter("a?", model=m, column="text", use_cascade=False)
    q_memo = Optimizer(
        OptimizerConfig(reorder=False, pushdown=False, cascade=False)
    ).optimize(q_plain)

    cost_plain = plan_cost(q_plain, cost_model, est).total_usd
    cost_memo = plan_cost(q_memo, cost_model, est).total_usd
    assert q_memo.ops[0].memoize is True
    assert cost_memo < cost_plain
    assert cost_memo == cost_plain * 0.1  # only 1 of 10 distinct inputs billed


def test_ablation_experiment_orders_savings():
    import experiments.run_ablations as ab

    cost_model = CostModel.from_yaml()
    table = ab.make_workload(n_dupe=4)
    q = ab.build_query(table)
    est = SelectivityEstimator(table, sample_size=1000)

    costs = {
        name: plan_cost(Optimizer(cfg).optimize(q), cost_model, est).total_usd
        for name, cfg in ab.CONFIGS.items()
    }
    # Full optimization is cheapest; disabling any pass costs more; naive is most.
    assert costs["full"] <= min(costs["-reorder"], costs["-pushdown"], costs["-memoize"])
    assert max(costs["-reorder"], costs["-pushdown"], costs["-memoize"]) <= costs["none (naive)"]


def test_execute_plan_with_map_and_memoize():
    t = _table({"a": 1}, n_each=5)  # duplicates → memoize path
    labeler = MockModel("gpt-4o", rule=lambda p: "positive")
    q = t.query().map("label it", output_column="lab", model=labeler, column="text")
    out = q.collect(optimize=True)  # optimizer marks memoize; physical wraps in cache
    assert "lab" in out.columns
    assert list(out["lab"]) == ["positive"] * 5
    assert out.provenance("lab").operator == "sem_map"


def test_explain_renders_plan():
    t = _table({"a": 4})
    q = t.query().filter("about a?", model=KeywordFilterModel("m", "a"), column="text")
    text = q.explain(optimize=True, sample_size=10)
    assert "Query plan" in text
    assert "Filter[" in text
    assert "estimated total cost" in text
