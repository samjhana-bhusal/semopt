"""Cost-based query optimizer (FR-8).

Four passes, each independently toggleable via :class:`OptimizerConfig` so the ablation
(eval plot P5) can attribute cost savings to each:

* **pushdown** (FR-8.2) — move filters toward the source, past projections/maps they
  don't depend on, so selective predicates shrink the row set before expensive ops.
* **reorder** (FR-8.1) — order each run of adjacent filters by ``cost / (1 − selectivity)``
  ascending (cheap, highly-selective predicates first). Selectivity is estimated from a
  cheap-tier sample (see :mod:`semopt.planner.selectivity`).
* **memoize** (FR-8.3/FR-9) — mark LLM ops whose inputs contain duplicates so execution
  wraps them in the LLM cache.
* **cascade** (FR-8.4) — attach a calibrated cascade to LLM filters requesting one, via a
  user-supplied ``cascade_factory`` (calibration needs labels, so the factory is the
  injection point).

Cost estimation (:func:`plan_cost`) walks the pipeline tracking the running row count:
each filter multiplies it by its selectivity, so a filter's dollar cost depends on where
it sits — which is exactly what reordering exploits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from semopt.cost.cost_model import CostModel
from semopt.planner.logical import Filter, LogicalOp, MapOp, Project, Query
from semopt.planner.selectivity import SelectivityEstimator
from semopt.table import SemanticTable

CascadeFactory = Callable[[Filter], object]


@dataclass
class OptimizerConfig:
    """Toggles for each optimizer pass (all on by default)."""

    reorder: bool = True
    pushdown: bool = True
    memoize: bool = True
    cascade: bool = True
    sample_size: int = 50
    cascade_factory: CascadeFactory | None = None


@dataclass
class OpCost:
    op: LogicalOp
    input_rows: float
    per_row_usd: float
    op_usd: float
    selectivity: float


@dataclass
class PlanCost:
    total_usd: float
    per_op: list[OpCost] = field(default_factory=list)


def op_per_row_cost(op: LogicalOp, cost_model: CostModel) -> float:
    """Estimated dollar cost of applying ``op`` to a single row."""
    if not op.is_llm:
        return 0.0
    model_id = None
    if isinstance(op, Filter):
        if op.cascade is not None:
            # Cascade: cheap tier runs on every row; a fraction escalates. Without the
            # calibrated escalation rate we bound below by the cheap tier's per-row cost.
            model_id = op.cascade.tiers[0].model.model_id
        elif op.model is not None:
            model_id = op.model.model_id
    elif isinstance(op, MapOp) and op.model is not None:
        model_id = op.model.model_id
    if model_id is None or model_id not in cost_model:
        return 0.0
    avg_out = 4.0 if isinstance(op, Filter) else 32.0
    dollars, _ = cost_model.estimate_operator_cost(
        model_id, 1, avg_tokens_in=64.0, avg_tokens_out=avg_out
    )
    return dollars


def plan_cost(
    query: Query,
    cost_model: CostModel,
    estimator: SelectivityEstimator,
) -> PlanCost:
    """Estimate total plan cost, tracking how each filter shrinks the row set."""
    rows = float(len(query.source))
    total = 0.0
    per_op: list[OpCost] = []
    for op in query.ops:
        per_row = op_per_row_cost(op, cost_model)
        op_usd = per_row * rows
        # A memoized LLM op only pays for distinct inputs (FR-8.3/FR-9).
        if op.is_llm and isinstance(op, Filter | MapOp) and op.memoize:
            op_usd *= estimator.unique_fraction(op)
        sel = 1.0
        if isinstance(op, Filter):
            sel = estimator.estimate(op).selectivity
        per_op.append(OpCost(op, rows, per_row, op_usd, sel))
        total += op_usd
        rows *= sel
    return PlanCost(total_usd=total, per_op=per_op)


class Optimizer:
    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config or OptimizerConfig()
        self._cost_model = CostModel.from_yaml()

    def optimize(self, query: Query) -> Query:
        estimator = SelectivityEstimator(
            query.source, sample_size=self.config.sample_size
        )
        ops = list(query.ops)
        if self.config.pushdown:
            ops = self._pushdown(ops)
        if self.config.reorder:
            ops = self._reorder(ops, estimator)
        if self.config.memoize:
            ops = self._insert_memoization(ops, query.source)
        if self.config.cascade:
            ops = self._insert_cascades(ops)
        return replace(query, ops=tuple(ops))

    # ------------------------------------------------------------- pushdown
    @staticmethod
    def _can_swap(flt: Filter, left: LogicalOp) -> bool:
        """True if ``flt`` (currently right of ``left``) may move before ``left``."""
        if isinstance(left, Filter):
            # Filters are mutually independent (none produces columns) — reordering is
            # handled by the dedicated pass, so leave adjacent filters in place here.
            return False
        if flt.reads_all:
            # Renders the whole row: unsafe to move past anything that drops (Project) or
            # adds (map ``produces``) columns, since either changes the rendered input.
            return not isinstance(left, Project) and not left.produces
        # Specific reads: unsafe if the filter needs a column ``left`` produces.
        if flt.reads & left.produces:
            return False
        # Past a projection, safe iff the filter's columns survive the projection.
        if isinstance(left, Project):
            return flt.reads <= set(left.columns)
        return True

    def _pushdown(self, ops: list[LogicalOp]) -> list[LogicalOp]:
        ops = list(ops)
        # Bubble each filter leftward past legal neighbors (stable for equal positions).
        for i in range(len(ops)):
            if not isinstance(ops[i], Filter):
                continue
            j = i
            while j > 0 and self._can_swap(ops[j], ops[j - 1]):  # type: ignore[arg-type]
                ops[j - 1], ops[j] = ops[j], ops[j - 1]
                j -= 1
        return ops

    # -------------------------------------------------------------- reorder
    def _reorder(
        self, ops: list[LogicalOp], estimator: SelectivityEstimator
    ) -> list[LogicalOp]:
        out: list[LogicalOp] = []
        i = 0
        n = len(ops)
        while i < n:
            if isinstance(ops[i], Filter):
                j = i
                while j < n and isinstance(ops[j], Filter):
                    j += 1
                run = [op for op in ops[i:j] if isinstance(op, Filter)]
                run.sort(key=lambda f: self._rank(f, estimator))
                out.extend(run)
                i = j
            else:
                out.append(ops[i])
                i += 1
        return out

    def _rank(self, flt: Filter, estimator: SelectivityEstimator) -> float:
        """Ordering key: cost / (1 − selectivity), ascending. Cheap + selective first."""
        sel = estimator.estimate(flt).selectivity
        cost = op_per_row_cost(flt, self._cost_model)
        reject = 1.0 - sel
        if reject <= 1e-9:
            return float("inf")  # keeps nothing back → run last
        return cost / reject

    # ------------------------------------------------------------ memoize
    @staticmethod
    def _has_duplicate_inputs(op: LogicalOp, table: SemanticTable) -> bool:
        df = table.to_pandas()
        if len(df) == 0:
            return False
        if op.reads_all:
            keyed = df.astype(str).agg("|".join, axis=1)
        else:
            cols = [c for c in op.reads if c in df.columns]
            if not cols:
                return False
            keyed = df[cols].astype(str).agg("|".join, axis=1)
        return bool(keyed.duplicated().any())

    def _insert_memoization(
        self, ops: list[LogicalOp], table: SemanticTable
    ) -> list[LogicalOp]:
        out: list[LogicalOp] = []
        for op in ops:
            if op.is_llm and isinstance(op, Filter | MapOp) and self._has_duplicate_inputs(
                op, table
            ):
                out.append(replace(op, memoize=True))
            else:
                out.append(op)
        return out

    # ------------------------------------------------------------ cascade
    def _insert_cascades(self, ops: list[LogicalOp]) -> list[LogicalOp]:
        factory = self.config.cascade_factory
        if factory is None:
            return ops
        out: list[LogicalOp] = []
        for op in ops:
            if (
                isinstance(op, Filter)
                and op.use_cascade
                and op.cascade is None
                and op.model is not None
            ):
                cascade = factory(op)
                out.append(replace(op, cascade=cascade))  # type: ignore[arg-type]
            else:
                out.append(op)
        return out
