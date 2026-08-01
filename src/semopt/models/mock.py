"""Deterministic mock backends for tests and offline demos.

These let the whole engine — operators, cache, cost accounting, and (later) the
cascade — run and be tested without MLX, an API key, or network access. They are
*deterministic* given their inputs so cached-replay and property tests are stable
(NFR-2).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from semopt.models.base import Model, ModelResponse


def _stable_unit(text: str) -> float:
    """Map a string to a stable pseudo-value in [0, 1) via its SHA-256 digest."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


class MockModel(Model):
    """A model whose answer is decided by a user-supplied rule.

    ``rule(prompt) -> str`` computes the answer. ``confidence`` (default derived from
    the prompt hash) fixes a deterministic per-call confidence so the confidence layer
    can be tested end-to-end. When ``supports_logprobs`` is True, the returned response
    carries a two-class logprob distribution consistent with ``confidence``.
    """

    def __init__(
        self,
        model_id: str,
        rule: Callable[[str], str],
        *,
        supports_logprobs: bool = True,
        confidence: Callable[[str], float] | None = None,
    ) -> None:
        self.model_id = model_id
        self.supports_logprobs = supports_logprobs
        self._rule = rule
        self._confidence = confidence

    def predict(
        self,
        prompt: str,
        *,
        examples: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> ModelResponse:
        import math

        value = self._rule(prompt)
        conf = self._confidence(prompt) if self._confidence else 0.5 + 0.5 * _stable_unit(prompt)
        conf = min(max(conf, 1e-6), 1 - 1e-6)

        logprobs: dict[str, float] | None = None
        if self.supports_logprobs:
            logprobs = {value: math.log(conf), f"~{value}": math.log(1 - conf)}

        tokens_in = max(1, len(prompt) // 4)
        tokens_out = max(1, len(value) // 4)
        return ModelResponse(
            value=value,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            wall_ms=0.0,
            logprobs=logprobs,
            model_id=self.model_id,
        )


class LookupModel(Model):
    """A deterministic backend returning a pre-scripted ``(value, confidence)`` per prompt.

    The confidence is emitted as a two-class logprob distribution so the confidence layer
    (FR-5.1) recovers exactly the scripted value. Used to build reproducible synthetic
    workloads for coverage experiments where the per-row model behavior must be fixed.
    Unknown prompts fall back to ``default``.
    """

    supports_logprobs = True

    def __init__(
        self,
        model_id: str,
        table: dict[str, tuple[str, float]],
        *,
        default: tuple[str, float] = ("no", 0.5),
    ) -> None:
        self.model_id = model_id
        self._table = table
        self._default = default

    def predict(
        self,
        prompt: str,
        *,
        examples: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> ModelResponse:
        import math

        value, conf = self._table.get(prompt, self._default)
        conf = min(max(conf, 1e-6), 1 - 1e-6)
        other = "no" if value == "yes" else "yes"
        logprobs = {value: math.log(conf), other: math.log(1 - conf)}
        return ModelResponse(
            value=value,
            tokens_in=max(1, len(prompt) // 4),
            tokens_out=1,
            wall_ms=0.0,
            logprobs=logprobs,
            model_id=self.model_id,
        )


def _row_body(prompt: str) -> str:
    """Return the row text an operator appended after the ``---`` delimiter.

    Operators render prompts as ``<instruction>\\n\\n---\\n<row>`` (see
    :meth:`semopt.operators.base.Operator.build_prompt`). A content-driven mock must look
    only at the row, not the instruction, otherwise a keyword in the question leaks into
    every decision.
    """
    _, sep, body = prompt.rpartition("\n---\n")
    return body if sep else prompt


class KeywordFilterModel(MockModel):
    """A boolean model that answers ``"yes"`` iff a keyword appears in the *row body*.

    Useful for exercising ``sem_filter`` deterministically: the ground truth is a pure
    function of the row text, so tests can assert exact selectivity. Only the portion of
    the prompt after the operator's ``---`` delimiter is inspected, so a keyword present
    in the instruction does not force every row to match.
    """

    def __init__(self, model_id: str, keyword: str, *, supports_logprobs: bool = True) -> None:
        kw = keyword.lower()

        def rule(prompt: str) -> str:
            return "yes" if kw in _row_body(prompt).lower() else "no"

        def confidence(prompt: str) -> float:
            # More confident the more times the keyword appears (or is absent).
            count = _row_body(prompt).lower().count(kw)
            return 0.6 + 0.35 * _stable_unit(prompt) + 0.04 * min(count, 3)

        super().__init__(
            model_id, rule, supports_logprobs=supports_logprobs, confidence=confidence
        )
