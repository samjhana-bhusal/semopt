"""sem_rank — top-k rows by LLM-judged relevance to a query (FR-2.5).

Two methods, because LLM comparisons are not guaranteed transitive (A>B, B>C, C>A is
possible), which breaks a plain comparison sort:

* **pointwise** (default) — score each row's relevance independently on a 0–10 scale, then
  sort. O(n) calls and transitivity-proof by construction (a total order on scalars).
* **pairwise** — ask the LLM to pick the more relevant of each pair, then aggregate with
  **Copeland** scores (number of pairwise wins). O(n²) calls but robust to intransitivity:
  a cycle just leaves the tied rows with equal win counts rather than corrupting the order.

Returns the top-``k`` rows with a ``rank_score`` column, highest first.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd

from semopt.cost.cost_model import CostModel
from semopt.operators.base import Operator
from semopt.table import ColumnProvenance, SemanticTable

if TYPE_CHECKING:
    from semopt.models.base import Model

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(text: str, *, lo: float = 0.0, hi: float = 10.0) -> float:
    """Extract the first number from a scoring response, clamped to ``[lo, hi]``."""
    m = _NUM.search(text)
    if not m:
        return lo
    return max(lo, min(hi, float(m.group(0))))


class SemRank(Operator):
    name = "sem_rank"

    def __init__(
        self,
        prompt: str,
        *,
        k: int,
        method: str = "pointwise",
        target_accuracy: float = 0.90,
        budget: float | None = None,
        column: str | None = None,
    ) -> None:
        super().__init__(
            prompt, target_accuracy=target_accuracy, budget=budget, column=column
        )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if method not in ("pointwise", "pairwise"):
            raise ValueError(f"method must be 'pointwise' or 'pairwise', got {method!r}")
        self.k = k
        self.method = method

    # ---------------------------------------------------------------- scoring
    def _pointwise_prompt(self, row: pd.Series) -> str:
        instruction = (
            f"{self.prompt}\nRate the relevance from 0 (irrelevant) to 10 (most "
            f"relevant). Answer with only the number."
        )
        return f"{instruction}\n\n---\n{self.render_row(row)}"

    def _pairwise_prompt(self, a: pd.Series, b: pd.Series) -> str:
        instruction = (
            f"{self.prompt}\nWhich item is more relevant? "
            f"Answer strictly 'A' or 'B'."
        )
        a_txt = self.render_row(a)
        b_txt = self.render_row(b)
        return f"{instruction}\n\n--- A ---\n{a_txt}\n\n--- B ---\n{b_txt}"

    def execute(self, table: SemanticTable, model: Model) -> SemanticTable:
        cost_model = CostModel.from_yaml()
        has_cost = model.model_id in cost_model
        df = table.to_pandas()
        n = len(df)

        acc = {"cost": 0.0, "tin": 0, "tout": 0}

        def call(prompt: str, max_tokens: int) -> str:
            resp = model.predict(prompt, max_tokens=max_tokens)
            acc["tin"] += resp.tokens_in
            acc["tout"] += resp.tokens_out
            if has_cost:
                acc["cost"] += cost_model.call_cost(
                    model.model_id, resp.tokens_in, resp.tokens_out
                )
            return resp.value

        if self.method == "pointwise":
            scores = [parse_score(call(self._pointwise_prompt(df.iloc[i]), 8)) for i in range(n)]
        else:
            scores = self._pairwise_copeland(df, call)

        order = sorted(range(n), key=lambda i: scores[i], reverse=True)[: self.k]
        top_df = df.iloc[order].copy().reset_index(drop=True)
        top_df["rank_score"] = [scores[i] for i in order]

        prov = ColumnProvenance(
            operator=self.name,
            model_id=model.model_id,
            cost_usd=acc["cost"],
            tokens_in=int(acc["tin"]),
            tokens_out=int(acc["tout"]),
            detail=(
                f"{self.method} rank: top {self.k} of {n}; alpha={self.alpha:.2f}"
            ),
        )
        provenance = {c: prov for c in top_df.columns}
        return SemanticTable(top_df, provenance)

    def _pairwise_copeland(self, df: pd.DataFrame, call) -> list[float]:  # type: ignore[no-untyped-def]
        """Copeland aggregation: each row's score is its count of pairwise wins.

        Robust to intransitive judgments — a 3-cycle contributes one win to each member,
        leaving them tied rather than producing an inconsistent sort.
        """
        n = len(df)
        wins = [0.0] * n
        for i in range(n):
            for j in range(i + 1, n):
                choice = call(self._pairwise_prompt(df.iloc[i], df.iloc[j]), 4).strip().upper()
                if choice.startswith("A"):
                    wins[i] += 1
                elif choice.startswith("B"):
                    wins[j] += 1
                else:  # unparseable → split the point (tie)
                    wins[i] += 0.5
                    wins[j] += 0.5
        return wins
