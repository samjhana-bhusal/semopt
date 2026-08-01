"""sem_extract — extract structured fields from each row (FR-2.3).

Given a list of field names, the LLM returns a value per field per row; each field becomes
a new column. The model is prompted to answer as a JSON object; parsing is tolerant —
a fenced ```json block, a bare object, or ``field: value`` lines all work, and missing
fields fall back to the empty string rather than raising, so one malformed row cannot
sink a whole extraction.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from semopt.cost.cost_model import CostModel
from semopt.operators.base import Operator
from semopt.table import ColumnProvenance

if TYPE_CHECKING:
    import pandas as pd

    from semopt.models.base import Model
    from semopt.table import SemanticTable

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_fields(text: str, fields: list[str]) -> dict[str, str]:
    """Best-effort parse of an LLM extraction response into ``{field: value}``.

    Tries, in order: a fenced JSON block, a bare ``{...}`` object, then ``field: value``
    lines. Always returns every requested field (empty string when absent).
    """
    obj: dict[str, object] | None = None

    match = _JSON_BLOCK.search(text) or _BARE_OBJECT.search(text)
    if match:
        candidate = match.group(1) if match.re is _JSON_BLOCK else match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                obj = parsed
        except (json.JSONDecodeError, ValueError):
            obj = None

    result: dict[str, str] = {}
    if obj is not None:
        lowered = {str(k).strip().lower(): v for k, v in obj.items()}
        for f in fields:
            val = lowered.get(f.strip().lower(), "")
            result[f] = "" if val is None else str(val).strip()
        return result

    # Fallback: line-oriented "field: value" scraping.
    line_map: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            line_map[key.strip().lower()] = val.strip()
    for f in fields:
        result[f] = line_map.get(f.strip().lower(), "")
    return result


class SemExtract(Operator):
    name = "sem_extract"

    def __init__(
        self,
        prompt: str,
        *,
        fields: list[str],
        target_accuracy: float = 0.90,
        budget: float | None = None,
        column: str | None = None,
    ) -> None:
        super().__init__(
            prompt, target_accuracy=target_accuracy, budget=budget, column=column
        )
        if not fields:
            raise ValueError("sem_extract needs at least one field")
        self.fields = fields

    def build_prompt(self, row: pd.Series) -> str:
        field_list = ", ".join(self.fields)
        instruction = (
            f"{self.prompt}\n"
            f"Extract these fields as a JSON object: {field_list}.\n"
            f"Respond with only the JSON object."
        )
        return f"{instruction}\n\n---\n{self.render_row(row)}"

    def execute(self, table: SemanticTable, model: Model) -> SemanticTable:
        cost_model = CostModel.from_yaml()
        has_cost = model.model_id in cost_model

        df = table.to_pandas()
        columns: dict[str, list[str]] = {f: [] for f in self.fields}
        total_cost = 0.0
        total_in = 0
        total_out = 0
        budget_hit = False

        for _, row in df.iterrows():
            if budget_hit:
                for f in self.fields:
                    columns[f].append("")
                continue
            resp = model.predict(self.build_prompt(row))
            parsed = parse_fields(resp.value, self.fields)
            for f in self.fields:
                columns[f].append(parsed[f])
            total_in += resp.tokens_in
            total_out += resp.tokens_out
            if has_cost:
                total_cost += cost_model.call_cost(
                    model.model_id, resp.tokens_in, resp.tokens_out
                )
            if self.budget is not None and total_cost > self.budget:
                budget_hit = True

        # Split the operator's cost across the produced columns so totals don't inflate.
        n_fields = len(self.fields)
        provenance = {
            f: ColumnProvenance(
                operator=self.name,
                model_id=model.model_id,
                cost_usd=total_cost / n_fields,
                tokens_in=total_in // n_fields,
                tokens_out=total_out // n_fields,
                detail=f"extracted '{f}' from {len(df)} rows; alpha={self.alpha:.2f}",
            )
            for f in self.fields
        }
        return table.with_columns(
            {f: columns[f] for f in self.fields}, provenance
        )
