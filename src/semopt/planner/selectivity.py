"""Selectivity estimation for cost-based reordering (FR-8.1).

The optimizer reorders filters so that cheap, highly-selective predicates run first and
shrink the row set before expensive ones. That needs an estimate of each filter's
*selectivity* — the fraction of rows it keeps. We estimate it by running the filter's
cheap tier over a small sample of the source rows and measuring the pass rate.

Estimates are cached per (filter prompt, column) so a filter is only sampled once.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from semopt.operators.filter import parse_bool
from semopt.planner.logical import Filter, LogicalOp
from semopt.table import SemanticTable


@dataclass
class SelectivityEstimate:
    selectivity: float  # fraction of rows kept, in [0, 1]
    n_sampled: int
    source: str  # "sampled" or "default"


class SelectivityEstimator:
    """Estimates filter selectivity from cheap-tier runs on a sample (FR-8.1)."""

    def __init__(
        self,
        table: SemanticTable,
        *,
        sample_size: int = 50,
        seed: int = 42,
        default_selectivity: float = 0.5,
    ) -> None:
        self._table = table
        self._sample_size = sample_size
        self._seed = seed
        self._default = default_selectivity
        self._cache: dict[tuple[str, str | None], SelectivityEstimate] = {}
        self._unique_cache: dict[frozenset[str] | None, float] = {}

    def _sample_df(self) -> pd.DataFrame:
        df = self._table.to_pandas()
        n = min(self._sample_size, len(df))
        if n == 0:
            return df
        return df.sample(n=n, random_state=self._seed)

    def estimate(self, flt: Filter) -> SelectivityEstimate:
        key = (flt.prompt, flt.column)
        if key in self._cache:
            return self._cache[key]

        # A model is required to sample; a cascade's cheap tier is used if present.
        model = flt.model
        if model is None and flt.cascade is not None:
            model = flt.cascade.tiers[0].model
        if model is None:
            est = SelectivityEstimate(self._default, 0, "default")
            self._cache[key] = est
            return est

        sample = self._sample_df()
        if len(sample) == 0:
            est = SelectivityEstimate(self._default, 0, "default")
            self._cache[key] = est
            return est

        instruction = f"{flt.prompt}\nAnswer strictly 'yes' or 'no'."
        kept = 0
        for _, row in sample.iterrows():
            body = (
                f"{flt.column}: {row[flt.column]}"
                if flt.column is not None
                else "\n".join(f"{c}: {v}" for c, v in row.items())
            )
            prompt = f"{instruction}\n\n---\n{body}"
            resp = model.predict(prompt, max_tokens=4)
            if parse_bool(resp.value):
                kept += 1

        sel = kept / len(sample)
        est = SelectivityEstimate(sel, len(sample), "sampled")
        self._cache[key] = est
        return est

    def unique_fraction(self, op: LogicalOp) -> float:
        """Fraction of *distinct* rendered inputs for ``op`` — how much memoization helps.

        A memoized op only issues one backend call per distinct input, so its effective
        cost scales by this fraction (FR-8.3/FR-9). Computed over the op's read columns
        (or all columns for whole-row ops) on the full source table.
        """
        cache_key = None if op.reads_all else frozenset(op.reads)
        if cache_key in self._unique_cache:
            return self._unique_cache[cache_key]

        df = self._table.to_pandas()
        if len(df) == 0:
            frac = 1.0
        else:
            if op.reads_all:
                keyed = df.astype(str).agg("|".join, axis=1)
            else:
                cols = [c for c in op.reads if c in df.columns]
                keyed = df[cols].astype(str).agg("|".join, axis=1) if cols else None
            frac = 1.0 if keyed is None else keyed.nunique() / len(df)
        self._unique_cache[cache_key] = frac
        return frac
