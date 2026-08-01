"""sem_filter — LLM boolean predicate (FR-2.1).

Runs the model once per row, parses a yes/no answer, and keeps the rows judged true.
Single-model in P0; the cascade wraps this later. Cost is accumulated from the actual
token counts each call reports, via the :class:`~semopt.cost.cost_model.CostModel`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from semopt.cost.cost_model import CostModel
from semopt.operators.base import Operator
from semopt.table import ColumnProvenance

if TYPE_CHECKING:
    from semopt.models.base import Model
    from semopt.table import SemanticTable

# Single-char aliases ("y"/"n"/"1"/"0") are matched only as the leading token, never
# as substrings, so words like "maybe" don't spuriously read as true.
_TRUE_FIRST = {"yes", "true", "1", "y", "complain", "complaining", "match"}
_FALSE_FIRST = {"no", "false", "0", "n"}
# Multi-char words safe to match anywhere in the answer.
_TRUE_ANYWHERE = {"yes", "true", "complain", "complaining", "match"}


def _strip_punct(token: str) -> str:
    return token.strip(".,:;!?\"'()[]")


def parse_bool(value: str) -> bool:
    """Interpret a model's free-text answer as a boolean.

    Looks at the first whitespace-delimited token (lowercased, de-punctuated); if that
    is inconclusive, scans the answer for a whole true-word. Ambiguous answers default
    to ``False`` (conservative: an unparseable predicate does not keep the row).
    """
    text = value.strip().lower()
    if not text:
        return False
    first = _strip_punct(text.split()[0])
    if first in _TRUE_FIRST:
        return True
    if first in _FALSE_FIRST:
        return False
    tokens = {_strip_punct(t) for t in text.split()}
    return bool(tokens & _TRUE_ANYWHERE)


class SemFilter(Operator):
    name = "sem_filter"

    def execute(self, table: SemanticTable, model: Model) -> SemanticTable:
        cost_model = CostModel.from_yaml()
        has_cost = model.model_id in cost_model

        df = table.to_pandas()
        mask: list[bool] = []
        total_cost = 0.0
        total_in = 0
        total_out = 0

        instruction = f"{self.prompt}\nAnswer strictly 'yes' or 'no'."
        for _, row in df.iterrows():
            prompt = self._render(instruction, row)
            resp = model.predict(prompt, max_tokens=4)
            mask.append(parse_bool(resp.value))
            total_in += resp.tokens_in
            total_out += resp.tokens_out
            if has_cost:
                total_cost += cost_model.call_cost(
                    model.model_id, resp.tokens_in, resp.tokens_out
                )
            if self.budget is not None and total_cost > self.budget:
                # Budget exhausted: keep remaining rows unevaluated (conservative: drop).
                remaining = len(df) - len(mask)
                mask.extend([False] * remaining)
                break

        prov = ColumnProvenance(
            operator=self.name,
            model_id=model.model_id,
            cost_usd=total_cost,
            tokens_in=total_in,
            tokens_out=total_out,
            detail=f"kept {sum(mask)}/{len(mask)} rows; alpha={self.alpha:.2f}",
        )
        return table.select_rows(mask, prov)

    def _render(self, instruction: str, row) -> str:  # type: ignore[no-untyped-def]
        body = self.render_row(row)
        return f"{instruction}\n\n---\n{body}"


class SemFilterCascade(Operator):
    """sem_filter run through a pre-calibrated confidence cascade (FR-6, FR-8.4).

    Unlike :class:`SemFilter` (single model), this dispatches each row through a
    :class:`~semopt.cascade.cascade.Cascade` whose thresholds were derived by conformal
    calibration (FR-7). Cost accumulates across whichever tiers each row actually used.
    The cascade must already be calibrated; calibration needs labels and is a separate
    step (see ``experiments/run_coverage.py``).
    """

    name = "sem_filter_cascade"

    def execute_with_cascade(self, table: SemanticTable, cascade) -> SemanticTable:  # type: ignore[no-untyped-def]
        df = table.to_pandas()
        instruction = f"{self.prompt}\nAnswer strictly 'yes' or 'no'."
        prompts = [self._render(instruction, row) for _, row in df.iterrows()]
        results = cascade.run(prompts)

        mask = [parse_bool(r.value) for r in results]
        total_cost = sum(r.total_cost_usd for r in results)
        total_in = sum(r.tokens_in for r in results)
        total_out = sum(r.tokens_out for r in results)

        n = max(len(results), 1)
        mean_tier = sum(r.final_tier for r in results) / n
        prov = ColumnProvenance(
            operator=self.name,
            model_id="+".join(t.model.model_id for t in cascade.tiers),
            cost_usd=total_cost,
            tokens_in=total_in,
            tokens_out=total_out,
            detail=(
                f"kept {sum(mask)}/{len(mask)} rows; alpha={self.alpha:.2f}; "
                f"mean_tier={mean_tier:.2f}"
            ),
        )
        return table.select_rows(mask, prov)

    def execute(self, table: SemanticTable, model: Model) -> SemanticTable:
        raise NotImplementedError("use execute_with_cascade() with a calibrated Cascade")

    def _render(self, instruction: str, row) -> str:  # type: ignore[no-untyped-def]
        body = self.render_row(row)
        return f"{instruction}\n\n---\n{body}"
