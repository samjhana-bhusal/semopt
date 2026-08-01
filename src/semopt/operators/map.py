"""sem_map — LLM produces a new value per row (FR-2.2).

Single-model in P0. Writes one new column (``output_column``) whose provenance records
the producing operator, model, and cost (FR-1.4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from semopt.cost.cost_model import CostModel
from semopt.operators.base import Operator
from semopt.table import ColumnProvenance

if TYPE_CHECKING:
    from semopt.models.base import Model
    from semopt.table import SemanticTable


class SemMap(Operator):
    name = "sem_map"

    def __init__(
        self,
        prompt: str,
        *,
        output_column: str,
        target_accuracy: float = 0.90,
        budget: float | None = None,
        column: str | None = None,
    ) -> None:
        super().__init__(
            prompt, target_accuracy=target_accuracy, budget=budget, column=column
        )
        self.output_column = output_column

    def execute(self, table: SemanticTable, model: Model) -> SemanticTable:
        cost_model = CostModel.from_yaml()
        has_cost = model.model_id in cost_model

        df = table.to_pandas()
        values: list[str] = []
        total_cost = 0.0
        total_in = 0
        total_out = 0
        budget_hit = False

        for _, row in df.iterrows():
            if budget_hit:
                values.append("")
                continue
            prompt = self.build_prompt(row)
            resp = model.predict(prompt)
            values.append(resp.value.strip())
            total_in += resp.tokens_in
            total_out += resp.tokens_out
            if has_cost:
                total_cost += cost_model.call_cost(
                    model.model_id, resp.tokens_in, resp.tokens_out
                )
            if self.budget is not None and total_cost > self.budget:
                budget_hit = True

        prov = ColumnProvenance(
            operator=self.name,
            model_id=model.model_id,
            cost_usd=total_cost,
            tokens_in=total_in,
            tokens_out=total_out,
            detail=f"mapped {len(values)} rows; alpha={self.alpha:.2f}",
        )
        return table.with_columns({self.output_column: values}, {self.output_column: prov})
