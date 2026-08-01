"""Physical plan: execute an optimized logical plan, and render ``explain()`` (FR-10.1).

Execution walks the (already optimized) op list in order, applying each to the running
:class:`~semopt.table.SemanticTable`. LLM ops marked ``memoize`` (by the optimizer's
memoization pass) are wrapped in a run-scoped in-memory :class:`~semopt.cache.memo.LLMCache`
so duplicate rows cost one backend call.

``explain_plan`` renders the plan as an indented operator list with per-op cost estimates;
with ``sample_size > 0`` it uses cheap-tier selectivity sampling so the row-count column
reflects estimated shrinkage, otherwise selectivity defaults to 0.5.
"""

from __future__ import annotations

from semopt.cache.memo import CachingModel, LLMCache
from semopt.cascade.cascade import Cascade, Tier
from semopt.cost.cost_model import CostModel
from semopt.models.base import Model
from semopt.planner.logical import Filter, LogicalOp, MapOp, Project, Query
from semopt.planner.optimizer import plan_cost
from semopt.planner.selectivity import SelectivityEstimator
from semopt.table import SemanticTable


def _wrap(model: Model, cache: LLMCache) -> CachingModel:
    return CachingModel(model, cache)


def _cascade_with_cache(cascade: Cascade, cache: LLMCache) -> Cascade:
    """Return a cascade whose tier models are cache-wrapped (thresholds preserved)."""
    tiers = [
        Tier(_wrap(t.model, cache), classification=t.classification, tau=t.tau)
        for t in cascade.tiers
    ]
    return Cascade(tiers, cost_model=cascade.cost_model, k_self_consistency=cascade.k)


def execute_plan(query: Query) -> SemanticTable:
    table = query.source
    cache = LLMCache(":memory:")  # run-scoped memoization store

    for op in query.ops:
        table = _execute_op(op, table, cache)
    return table


def _execute_op(op: LogicalOp, table: SemanticTable, cache: LLMCache) -> SemanticTable:
    if isinstance(op, Project):
        return table.project(list(op.columns))

    if isinstance(op, Filter):
        if op.cascade is not None:
            cascade = _cascade_with_cache(op.cascade, cache) if op.memoize else op.cascade
            return table.sem_filter_cascade(
                op.prompt,
                cascade=cascade,
                target_accuracy=op.target_accuracy,
                column=op.column,
            )
        assert op.model is not None
        model = _wrap(op.model, cache) if op.memoize else op.model
        return table.sem_filter(
            op.prompt,
            model=model,
            target_accuracy=op.target_accuracy,
            column=op.column,
        )

    if isinstance(op, MapOp):
        assert op.model is not None
        model = _wrap(op.model, cache) if op.memoize else op.model
        return table.sem_map(
            op.prompt,
            model=model,
            output_column=op.output_column,
            target_accuracy=op.target_accuracy,
            column=op.column,
        )

    raise TypeError(f"unknown logical op: {type(op).__name__}")


def explain_plan(query: Query, *, sample_size: int = 0) -> str:
    """Render the plan with estimated per-op cost and running row counts (FR-10.1)."""
    cost_model = CostModel.from_yaml()
    estimator = SelectivityEstimator(
        query.source, sample_size=max(sample_size, 1), default_selectivity=0.5
    )
    if sample_size <= 0:
        # Neutralize sampling: force the default selectivity without hitting any model.
        estimator._sample_size = 0  # noqa: SLF001 - intentional: skip sampling in explain

    pc = plan_cost(query, cost_model, estimator)

    lines = [
        f"Query plan  (source: {len(query.source)} rows, "
        f"{len(query.source.columns)} cols)",
        f"  estimated total cost: ${pc.total_usd:.6f}",
        "",
    ]
    for depth, oc in enumerate(pc.per_op):
        indent = "  " + "  " * depth + "└─ "
        sel = f"sel={oc.selectivity:.2f}" if isinstance(oc.op, Filter) else "sel=1.00"
        lines.append(
            f"{indent}{oc.op.label()}  "
            f"[in={oc.input_rows:.0f} rows, {sel}, "
            f"${oc.op_usd:.6f}]"
        )
    return "\n".join(lines)
