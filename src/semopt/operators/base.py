"""Operator base class (FR-2).

Each semantic operator takes a :class:`~semopt.table.SemanticTable` plus a
:class:`~semopt.models.base.Model` and returns a new table. In P0 operators run a
single model directly, one call per row; the cascade (FR-6) and optimizer (FR-8) slot
in later without changing this interface.

Prompt rendering: the user's natural-language ``prompt`` is combined with the row's
text. By default every string column is serialized as ``key: value`` lines; a specific
``column`` can be named to send only that field.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from semopt.models.base import Model
    from semopt.table import SemanticTable


class Operator(ABC):
    """Base class for all semantic operators."""

    name: str = "operator"

    def __init__(
        self,
        prompt: str,
        *,
        target_accuracy: float = 0.90,
        budget: float | None = None,
        column: str | None = None,
    ) -> None:
        if not 0.0 < target_accuracy < 1.0:
            raise ValueError(f"target_accuracy must be in (0, 1), got {target_accuracy}")
        if budget is not None and budget < 0:
            raise ValueError(f"budget must be non-negative or None, got {budget}")
        self.prompt = prompt
        self.target_accuracy = target_accuracy
        self.alpha = 1.0 - target_accuracy
        self.budget = budget
        self.column = column

    @abstractmethod
    def execute(self, table: SemanticTable, model: Model) -> SemanticTable:
        raise NotImplementedError

    # --------------------------------------------------------- prompt rendering
    def render_row(self, row: pd.Series) -> str:
        """Serialize a row into the text block appended to the operator prompt."""
        if self.column is not None:
            return f"{self.column}: {row[self.column]}"
        parts = []
        for col, val in row.items():
            parts.append(f"{col}: {val}")
        return "\n".join(parts)

    def build_prompt(self, row: pd.Series) -> str:
        return f"{self.prompt}\n\n---\n{self.render_row(row)}"
