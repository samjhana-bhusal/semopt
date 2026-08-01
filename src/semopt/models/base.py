"""Common LLM model interface (FR-3.1).

Every backend — MLX-local, OpenAI/Anthropic API, or the deterministic mock used in
tests — implements :class:`Model`. Confidence scoring (FR-5) and the cascade (FR-6)
depend only on this interface, never on a concrete backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelResponse:
    """The result of a single LLM call.

    ``logprobs`` maps a candidate string (e.g. ``"yes"`` / ``"no"``, or a generated
    token) to its log-probability, when the backend exposes them. It is ``None`` when
    the backend cannot provide token-level logprobs — confidence then falls back to
    self-consistency (FR-5.2 / FR-3.3).
    """

    value: str
    tokens_in: int
    tokens_out: int
    wall_ms: float
    logprobs: dict[str, float] | None = None
    model_id: str = ""
    raw: Any = field(default=None, repr=False, compare=False)


class Model(ABC):
    """Abstract LLM backend."""

    #: Stable identifier used for cost lookup and cache keying.
    model_id: str

    #: Whether this backend can return token-level logprobs (FR-3.2 / FR-3.3).
    supports_logprobs: bool = False

    @abstractmethod
    def predict(
        self,
        prompt: str,
        *,
        examples: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> ModelResponse:
        """Run the model on ``prompt`` and return a :class:`ModelResponse`.

        ``examples`` is an optional list of ``(input, output)`` few-shot pairs.
        """
        raise NotImplementedError
