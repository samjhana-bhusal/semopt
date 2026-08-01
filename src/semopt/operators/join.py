"""sem_join — semantic join of two tables (FR-2.4).

Row ``i`` of the left table joins row ``j`` of the right iff the LLM judges them a match
for ``on_prompt``. Two strategies:

* **nested_loop** — evaluate every left×right pair. Correct but O(n·m) LLM calls.
* **blocked** — only pairs sharing an (exact) blocking-key value are evaluated, cutting
  the candidate set the way a blocked-nested-loop join does. Approximate: true matches
  that disagree on the blocking key are missed (a recall/cost trade the caller controls).

Output columns are the union of both tables' columns; right-side name collisions get a
suffix. Cost accrues per evaluated pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from semopt.cost.cost_model import CostModel
from semopt.operators.base import Operator
from semopt.operators.filter import parse_bool
from semopt.table import ColumnProvenance, SemanticTable

if TYPE_CHECKING:
    from semopt.models.base import Model


class SemJoin(Operator):
    name = "sem_join"

    def __init__(
        self,
        on_prompt: str,
        *,
        how: str = "nested_loop",
        left_block: str | None = None,
        right_block: str | None = None,
        suffix: str = "_right",
        target_accuracy: float = 0.90,
        budget: float | None = None,
    ) -> None:
        super().__init__(on_prompt, target_accuracy=target_accuracy, budget=budget)
        if how not in ("nested_loop", "blocked"):
            raise ValueError(f"how must be 'nested_loop' or 'blocked', got {how!r}")
        if how == "blocked" and (left_block is None or right_block is None):
            raise ValueError("blocked join requires left_block and right_block")
        self.how = how
        self.left_block = left_block
        self.right_block = right_block
        self.suffix = suffix

    def execute(self, table: SemanticTable, model: Model) -> SemanticTable:
        raise NotImplementedError("use execute_join(left, right, model)")

    def _pair_prompt(self, lrow: pd.Series, rrow: pd.Series) -> str:
        left_txt = "\n".join(f"{c}: {v}" for c, v in lrow.items())
        right_txt = "\n".join(f"{c}: {v}" for c, v in rrow.items())
        instruction = f"{self.prompt}\nAnswer strictly 'yes' or 'no'."
        return f"{instruction}\n\n--- LEFT ---\n{left_txt}\n\n--- RIGHT ---\n{right_txt}"

    def _candidate_pairs(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> list[tuple[int, int]]:
        if self.how == "nested_loop":
            return [(i, j) for i in range(len(left)) for j in range(len(right))]
        # Blocked: group right rows by blocking key, pair only within matching blocks.
        assert self.left_block is not None and self.right_block is not None
        right_by_key: dict[object, list[int]] = {}
        for j in range(len(right)):
            key = right.iloc[j][self.right_block]
            right_by_key.setdefault(key, []).append(j)
        pairs: list[tuple[int, int]] = []
        for i in range(len(left)):
            key = left.iloc[i][self.left_block]
            for j in right_by_key.get(key, []):
                pairs.append((i, j))
        return pairs

    def execute_join(
        self, left: SemanticTable, right: SemanticTable, model: Model
    ) -> SemanticTable:
        cost_model = CostModel.from_yaml()
        has_cost = model.model_id in cost_model

        ldf = left.to_pandas()
        rdf = right.to_pandas()
        # Disambiguate colliding right columns up front.
        overlap = set(ldf.columns) & set(rdf.columns)
        rdf_renamed = rdf.rename(columns={c: f"{c}{self.suffix}" for c in overlap})

        pairs = self._candidate_pairs(ldf, rdf)
        joined_rows: list[dict[str, object]] = []
        total_cost = 0.0
        total_in = 0
        total_out = 0

        for i, j in pairs:
            resp = model.predict(self._pair_prompt(ldf.iloc[i], rdf.iloc[j]), max_tokens=4)
            total_in += resp.tokens_in
            total_out += resp.tokens_out
            if has_cost:
                total_cost += cost_model.call_cost(
                    model.model_id, resp.tokens_in, resp.tokens_out
                )
            if parse_bool(resp.value):
                merged = {**ldf.iloc[i].to_dict(), **rdf_renamed.iloc[j].to_dict()}
                joined_rows.append(merged)
            if self.budget is not None and total_cost > self.budget:
                break

        all_columns = list(ldf.columns) + list(rdf_renamed.columns)
        result_df = pd.DataFrame(joined_rows, columns=all_columns)

        prov = ColumnProvenance(
            operator=self.name,
            model_id=model.model_id,
            cost_usd=total_cost,
            tokens_in=total_in,
            tokens_out=total_out,
            detail=(
                f"{self.how} join: {len(pairs)} pairs evaluated, "
                f"{len(joined_rows)} matched; alpha={self.alpha:.2f}"
            ),
        )
        provenance = {c: prov for c in all_columns}
        return SemanticTable(result_df, provenance)
