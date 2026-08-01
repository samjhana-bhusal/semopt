"""Cost model (FR-4).

Loads the per-model cost table from ``costs.yaml`` and turns token counts / operator
sizes into dollar and latency estimates. Every experiment reports ``$`` spent (NFR-4);
this module is the single source of truth for converting tokens → dollars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_COSTS = Path(__file__).with_name("costs.yaml")


@dataclass(frozen=True)
class ModelCost:
    model_id: str
    usd_per_1m_input: float
    usd_per_1m_output: float
    ms_per_call: float
    tier: str

    def call_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Dollar cost of a single call with the given token counts."""
        return (
            tokens_in / 1_000_000 * self.usd_per_1m_input
            + tokens_out / 1_000_000 * self.usd_per_1m_output
        )


class CostModel:
    """The per-model cost table (FR-4.1) plus estimation helpers (FR-4.2)."""

    def __init__(self, costs: dict[str, ModelCost]) -> None:
        self._costs = costs

    @classmethod
    def from_yaml(cls, path: str | Path = _DEFAULT_COSTS) -> CostModel:
        with open(path) as fh:
            data = yaml.safe_load(fh)
        costs = {
            model_id: ModelCost(
                model_id=model_id,
                usd_per_1m_input=float(spec["usd_per_1m_input"]),
                usd_per_1m_output=float(spec["usd_per_1m_output"]),
                ms_per_call=float(spec["ms_per_call"]),
                tier=str(spec.get("tier", "unknown")),
            )
            for model_id, spec in data["models"].items()
        }
        return cls(costs)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._costs

    def get(self, model_id: str) -> ModelCost:
        if model_id not in self._costs:
            raise KeyError(
                f"no cost entry for model {model_id!r}; add it to costs.yaml (FR-4.1)"
            )
        return self._costs[model_id]

    def call_cost(self, model_id: str, tokens_in: int, tokens_out: int) -> float:
        return self.get(model_id).call_cost(tokens_in, tokens_out)

    def estimate_operator_cost(
        self,
        model_id: str,
        input_size: int,
        *,
        avg_tokens_in: float = 200.0,
        avg_tokens_out: float = 10.0,
    ) -> tuple[float, float]:
        """Return ``(expected_dollars, expected_ms)`` for running one LLM-invoking
        operator over ``input_size`` rows on ``model_id`` (FR-4.2).

        Token averages default to filter-shaped calls (long input, tiny output) and can
        be overridden once measured per operator.
        """
        mc = self.get(model_id)
        dollars = input_size * mc.call_cost(int(avg_tokens_in), int(avg_tokens_out))
        ms = input_size * mc.ms_per_call
        return dollars, ms
