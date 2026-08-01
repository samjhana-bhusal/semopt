"""Logical query plan and the lazy ``Query`` builder (FR-8, FR-10.1).

The P0/P1 operator API is *eager* — ``table.sem_filter(...)`` runs immediately. To
optimize (reorder, push down, insert cascades/memoization) we need to see the whole
pipeline before running it, so this module adds a *lazy* layer:

    q = (Query.from_table(reviews)
            .project(["text"])
            .filter("complains about shipping?", model=cheap)
            .filter("mentions a refund?", model=cheap))
    print(q.explain())      # optimized plan + cost estimate (FR-10.1)
    result = q.collect()    # optimize, then execute → SemanticTable

A plan is a ``Source`` plus an ordered list of unary :class:`LogicalOp` s (our operators
are unary in P2; ``sem_join`` arrives in P3). Each op declares the columns it *reads* and
*produces* so the optimizer can reorder and push it down without changing results.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semopt.cascade.cascade import Cascade
    from semopt.models.base import Model
    from semopt.table import SemanticTable


@dataclass(frozen=True)
class LogicalOp:
    """Base logical operator. Subclasses declare column reads/writes and LLM use."""

    #: Specific columns this op reads (used for pushdown safety).
    reads: frozenset[str] = field(default_factory=frozenset)
    #: If True, the op reads *every* column present at its position (column=None ops):
    #: it renders the whole row, so it cannot be pushed past ops that add/drop columns.
    reads_all: bool = False
    #: Columns this op adds.
    produces: frozenset[str] = field(default_factory=frozenset)
    #: Whether the op issues LLM calls (matters for cost, memoization, cascades).
    is_llm: bool = False

    def label(self) -> str:  # pragma: no cover - overridden
        return type(self).__name__


@dataclass(frozen=True)
class Project(LogicalOp):
    """Keep only ``columns`` (non-LLM, effectively free)."""

    columns: tuple[str, ...] = ()

    def label(self) -> str:
        return f"Project[{', '.join(self.columns)}]"


@dataclass(frozen=True)
class Filter(LogicalOp):
    """LLM predicate; keeps rows judged true (FR-2.1). Optionally cascaded (FR-8.4)."""

    prompt: str = ""
    model: Model | None = None
    cascade: Cascade | None = None
    target_accuracy: float = 0.90
    column: str | None = None
    use_cascade: bool = True
    memoize: bool = False

    def label(self) -> str:
        tag = "cascade" if self.cascade is not None else (
            "cascade?" if self.use_cascade else "single"
        )
        memo = "+memo" if self.memoize else ""
        short = self.prompt if len(self.prompt) <= 32 else self.prompt[:29] + "..."
        return f"Filter[{short!r} {tag}{memo}]"


@dataclass(frozen=True)
class MapOp(LogicalOp):
    """LLM map producing ``output_column`` (FR-2.2)."""

    prompt: str = ""
    output_column: str = "output"
    model: Model | None = None
    target_accuracy: float = 0.90
    column: str | None = None
    memoize: bool = False

    def label(self) -> str:
        memo = "+memo" if self.memoize else ""
        short = self.prompt if len(self.prompt) <= 32 else self.prompt[:29] + "..."
        return f"Map[{self.output_column} <- {short!r}{memo}]"


def _reads_for(prompt_column: str | None) -> tuple[frozenset[str], bool]:
    """Return ``(reads, reads_all)`` for a filter/map given its ``column`` argument."""
    if prompt_column is not None:
        return frozenset({prompt_column}), False
    return frozenset(), True


@dataclass(frozen=True)
class Query:
    """An immutable lazy pipeline: a source table plus an ordered op list."""

    source: SemanticTable
    ops: tuple[LogicalOp, ...] = ()

    @classmethod
    def from_table(cls, table: SemanticTable) -> Query:
        return cls(source=table, ops=())

    @property
    def source_columns(self) -> frozenset[str]:
        return frozenset(self.source.columns)

    def _append(self, op: LogicalOp) -> Query:
        return replace(self, ops=(*self.ops, op))

    def project(self, columns: list[str]) -> Query:
        return self._append(Project(columns=tuple(columns), reads=frozenset(columns)))

    def filter(
        self,
        prompt: str,
        *,
        model: Model | None = None,
        cascade: Cascade | None = None,
        target_accuracy: float = 0.90,
        column: str | None = None,
        use_cascade: bool = True,
    ) -> Query:
        if model is None and cascade is None:
            raise ValueError("filter needs either a model or a cascade")
        reads, reads_all = _reads_for(column)
        return self._append(
            Filter(
                prompt=prompt,
                model=model,
                cascade=cascade,
                target_accuracy=target_accuracy,
                column=column,
                use_cascade=use_cascade,
                is_llm=True,
                reads=reads,
                reads_all=reads_all,
            )
        )

    def map(
        self,
        prompt: str,
        *,
        output_column: str,
        model: Model,
        target_accuracy: float = 0.90,
        column: str | None = None,
    ) -> Query:
        reads, reads_all = _reads_for(column)
        return self._append(
            MapOp(
                prompt=prompt,
                output_column=output_column,
                model=model,
                target_accuracy=target_accuracy,
                column=column,
                is_llm=True,
                reads=reads,
                reads_all=reads_all,
                produces=frozenset({output_column}),
            )
        )

    # ------------------------------------------------------- optimize / run
    def optimize(self, config: object | None = None) -> Query:
        from semopt.planner.optimizer import Optimizer, OptimizerConfig

        cfg = config if isinstance(config, OptimizerConfig) else OptimizerConfig()
        return Optimizer(cfg).optimize(self)

    def explain(self, *, optimize: bool = True, sample_size: int = 0) -> str:
        from semopt.planner.physical import explain_plan

        plan = self.optimize() if optimize else self
        return explain_plan(plan, sample_size=sample_size)

    def collect(self, *, optimize: bool = True) -> SemanticTable:
        from semopt.planner.physical import execute_plan

        plan = self.optimize() if optimize else self
        return execute_plan(plan)
