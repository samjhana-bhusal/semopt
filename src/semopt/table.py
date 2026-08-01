"""SemanticTable — the user-facing table abstraction (FR-1).

Wraps a pandas DataFrame with:
  * immutable semantics — every operator returns a new table (FR-1.3);
  * multi-format loading — Parquet / CSV / JSON / list-of-dicts (FR-1.1);
  * pandas-compatible column access — ``table["col"]`` (FR-1.2);
  * per-column provenance — which operator produced it, and at what cost (FR-1.4).

The semantic operators (``sem_filter`` etc.) are thin methods that delegate to the
operator classes in :mod:`semopt.operators`, executed against an
:class:`~semopt.models.base.Model`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from semopt.models.base import Model


@dataclass(frozen=True)
class ColumnProvenance:
    """How a single column came to exist (FR-1.4).

    ``operator`` is ``"source"`` for columns loaded from disk; otherwise it is the
    name of the semantic operator that produced the column. ``cost_usd`` and the token
    counts are the *cumulative* cost attributed to producing this column.
    """

    operator: str
    model_id: str | None = None
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    detail: str = ""


class SemanticTable:
    """An immutable table of rows over which semantic operators run.

    Construct via the ``from_*`` classmethods rather than the initializer directly when
    loading external data. Operators return *new* tables; the underlying DataFrame is
    defensively copied so callers cannot mutate shared state.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        provenance: dict[str, ColumnProvenance] | None = None,
    ) -> None:
        self._df = df.reset_index(drop=True).copy()
        if provenance is None:
            provenance = {c: ColumnProvenance(operator="source") for c in self._df.columns}
        else:
            # Ensure every column has a provenance entry; default unknown ones to source.
            provenance = {
                c: provenance.get(c, ColumnProvenance(operator="source"))
                for c in self._df.columns
            }
        self._provenance: dict[str, ColumnProvenance] = provenance

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_parquet(cls, path: str | Path) -> SemanticTable:
        return cls(pd.read_parquet(path))

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: Any) -> SemanticTable:
        return cls(pd.read_csv(path, **kwargs))

    @classmethod
    def from_json(cls, path: str | Path) -> SemanticTable:
        """Load a JSON file that is either a list of records or ``{col: [values]}``."""
        with open(path) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return cls.from_records(data)
        return cls(pd.DataFrame(data))

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> SemanticTable:
        return cls(pd.DataFrame.from_records(records))

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self._df)

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self._df.shape
        return int(rows), int(cols)

    def __getitem__(self, key: str) -> pd.Series:
        """Pandas-compatible column access (FR-1.2)."""
        return self._df[key]

    def __contains__(self, key: str) -> bool:
        return key in self._df.columns

    def to_pandas(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame (immutability preserved)."""
        return self._df.copy()

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._df.head(n).copy()

    def provenance(self, column: str) -> ColumnProvenance:
        return self._provenance[column]

    def total_cost_usd(self) -> float:
        """Sum of costs attributed across all columns."""
        return float(sum(p.cost_usd for p in self._provenance.values()))

    # ------------------------------------------------- immutable derivations
    def with_columns(
        self,
        new_data: dict[str, Any],
        provenance: dict[str, ColumnProvenance],
    ) -> SemanticTable:
        """Return a new table with columns added/overwritten and provenance recorded."""
        df = self._df.copy()
        for name, values in new_data.items():
            df[name] = values
        merged = dict(self._provenance)
        merged.update(provenance)
        return SemanticTable(df, merged)

    def project(self, columns: list[str]) -> SemanticTable:
        """Return a new table with only ``columns`` (order preserved), provenance kept."""
        missing = [c for c in columns if c not in self._df.columns]
        if missing:
            raise KeyError(f"cannot project unknown columns: {missing}")
        df = self._df[columns].copy()
        prov = {c: self._provenance[c] for c in columns}
        return SemanticTable(df, prov)

    def select_rows(self, mask: Any, operator_provenance: ColumnProvenance) -> SemanticTable:
        """Return a new table keeping only rows where ``mask`` is truthy.

        Row selection does not create columns, so the operator's cost is folded into
        every existing column's provenance so it is not lost from ``total_cost_usd``.
        """
        df = self._df[list(mask)].reset_index(drop=True).copy()
        merged: dict[str, ColumnProvenance] = {}
        # Attribute the filter's cost once, to the first column, to avoid double counting.
        cost_added = False
        for col, prov in self._provenance.items():
            if col not in df.columns:
                continue
            if not cost_added:
                merged[col] = replace(
                    prov,
                    cost_usd=prov.cost_usd + operator_provenance.cost_usd,
                    tokens_in=prov.tokens_in + operator_provenance.tokens_in,
                    tokens_out=prov.tokens_out + operator_provenance.tokens_out,
                )
                cost_added = True
            else:
                merged[col] = prov
        return SemanticTable(df, merged)

    # ------------------------------------------------- semantic operators
    def sem_filter(
        self,
        prompt: str,
        *,
        model: Model,
        target_accuracy: float = 0.90,
        budget: float | None = None,
        column: str | None = None,
    ) -> SemanticTable:
        """Keep rows for which the LLM predicate is true (FR-2.1)."""
        from semopt.operators.filter import SemFilter

        return SemFilter(
            prompt, target_accuracy=target_accuracy, budget=budget, column=column
        ).execute(self, model)

    def query(self) -> Any:
        """Start a lazy, optimizable pipeline over this table (FR-8, FR-10.1).

        Returns a :class:`~semopt.planner.logical.Query`; chain ``.filter``/``.map``/
        ``.project``, then ``.explain()`` or ``.collect()``.
        """
        from semopt.planner.logical import Query

        return Query.from_table(self)

    def sem_filter_cascade(
        self,
        prompt: str,
        *,
        cascade: object,
        target_accuracy: float = 0.90,
        column: str | None = None,
    ) -> SemanticTable:
        """Keep rows for which a *calibrated cascade* judges the predicate true (FR-6)."""
        from semopt.operators.filter import SemFilterCascade

        return SemFilterCascade(
            prompt, target_accuracy=target_accuracy, column=column
        ).execute_with_cascade(self, cascade)

    def sem_map(
        self,
        prompt: str,
        *,
        model: Model,
        output_column: str,
        target_accuracy: float = 0.90,
        budget: float | None = None,
        column: str | None = None,
    ) -> SemanticTable:
        """Produce a new value per row via the LLM (FR-2.2)."""
        from semopt.operators.map import SemMap

        return SemMap(
            prompt,
            output_column=output_column,
            target_accuracy=target_accuracy,
            budget=budget,
            column=column,
        ).execute(self, model)

    def sem_extract(
        self,
        prompt: str,
        *,
        fields: list[str],
        model: Model,
        target_accuracy: float = 0.90,
        budget: float | None = None,
        column: str | None = None,
    ) -> SemanticTable:
        """Extract structured ``fields`` as new columns (FR-2.3)."""
        from semopt.operators.extract import SemExtract

        return SemExtract(
            prompt,
            fields=fields,
            target_accuracy=target_accuracy,
            budget=budget,
            column=column,
        ).execute(self, model)

    def sem_join(
        self,
        other: SemanticTable,
        on_prompt: str,
        *,
        model: Model,
        how: str = "nested_loop",
        left_block: str | None = None,
        right_block: str | None = None,
        suffix: str = "_right",
        target_accuracy: float = 0.90,
        budget: float | None = None,
    ) -> SemanticTable:
        """Semantic join with ``other``; keep pairs the LLM judges a match (FR-2.4)."""
        from semopt.operators.join import SemJoin

        return SemJoin(
            on_prompt,
            how=how,
            left_block=left_block,
            right_block=right_block,
            suffix=suffix,
            target_accuracy=target_accuracy,
            budget=budget,
        ).execute_join(self, other, model)

    def sem_rank(
        self,
        prompt: str,
        *,
        k: int,
        model: Model,
        method: str = "pointwise",
        target_accuracy: float = 0.90,
        budget: float | None = None,
        column: str | None = None,
    ) -> SemanticTable:
        """Return the top-``k`` rows by LLM-judged relevance (FR-2.5)."""
        from semopt.operators.rank import SemRank

        return SemRank(
            prompt,
            k=k,
            method=method,
            target_accuracy=target_accuracy,
            budget=budget,
            column=column,
        ).execute(self, model)

    def __repr__(self) -> str:
        return f"SemanticTable(rows={len(self)}, columns={self.columns})"
