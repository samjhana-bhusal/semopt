"""Confidence scoring for cascade escalation (FR-5).

Two paths, matching the operator kind:

* **Logprob confidence (FR-5.1)** — for classification-style operators (``sem_filter``,
  boolean ``sem_map``). Given the model's token logprobs over the answer position, the
  confidence is ``max(P(class))`` after renormalizing over the candidate class tokens.

* **Self-consistency (FR-5.2)** — for open-ended operators, or any backend that cannot
  expose logprobs (FR-3.3, e.g. the Anthropic API). Sample ``K`` completions at a
  non-zero temperature; the confidence is the fraction agreeing with the modal answer,
  with semantic equivalence decided by a normalizer (default: whitespace/case/punct
  normalization).

Neither score is itself the guarantee (FR-5.3) — it is the *input* to the conformal
calibration layer (FR-7), which turns raw confidences into a threshold with a coverage
guarantee.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from semopt.models.base import Model, ModelResponse

# Default candidate tokens for a yes/no predicate. Compared case-insensitively after
# stripping punctuation, so "Yes", " yes", "yes." all match.
DEFAULT_TRUE_TOKENS = frozenset({"yes", "true", "y"})
DEFAULT_FALSE_TOKENS = frozenset({"no", "false", "n"})

_NORM_WS = re.compile(r"\s+")
_NORM_PUNCT = re.compile(r"[.,:;!?\"'()\[\]]")


@dataclass(frozen=True)
class Confidence:
    """A prediction plus its calibrated-input confidence.

    ``value`` is the (possibly canonicalized) answer; ``score`` is in ``[0, 1]``;
    ``method`` records which path produced it, for tracing.
    """

    value: str
    score: float
    method: str


def _norm_token(tok: str) -> str:
    return _NORM_PUNCT.sub("", tok.strip().lower())


def logprob_confidence(
    response: ModelResponse,
    *,
    true_tokens: frozenset[str] = DEFAULT_TRUE_TOKENS,
    false_tokens: frozenset[str] = DEFAULT_FALSE_TOKENS,
) -> Confidence | None:
    """Confidence for a boolean answer from token logprobs (FR-5.1).

    Returns ``None`` when the response carries no logprobs or none of the top tokens
    match a class — the caller should then fall back to self-consistency.
    """
    if not response.logprobs:
        return None

    p_true = 0.0
    p_false = 0.0
    for tok, lp in response.logprobs.items():
        norm = _norm_token(tok)
        if norm in true_tokens:
            p_true += math.exp(lp)
        elif norm in false_tokens:
            p_false += math.exp(lp)

    total = p_true + p_false
    if total <= 0.0:
        return None

    p_true /= total
    p_false /= total
    if p_true >= p_false:
        return Confidence(value="yes", score=p_true, method="logprob")
    return Confidence(value="no", score=p_false, method="logprob")


def normalize_answer(text: str) -> str:
    """Canonicalize a free-text answer for semantic-equivalence comparison (FR-5.2)."""
    return _NORM_WS.sub(" ", _NORM_PUNCT.sub("", text.strip().lower())).strip()


def self_consistency_confidence(
    model: Model,
    prompt: str,
    *,
    k: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 256,
    base_seed: int = 0,
    normalizer: Callable[[str], str] = normalize_answer,
) -> Confidence:
    """Confidence via self-consistency sampling (FR-5.2).

    Draws ``k`` completions at ``temperature`` (fixed, incrementing seeds for
    reproducibility, NFR-2), groups them by ``normalizer``, and reports the modal
    answer with ``score = count(modal) / k``. The returned ``value`` is the first raw
    completion that fell in the modal group.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    raw: list[str] = []
    normed: list[str] = []
    for i in range(k):
        resp = model.predict(
            prompt, temperature=temperature, max_tokens=max_tokens, seed=base_seed + i
        )
        raw.append(resp.value)
        normed.append(normalizer(resp.value))

    counts = Counter(normed)
    modal_norm, modal_count = counts.most_common(1)[0]
    modal_value = next(raw[i] for i in range(k) if normed[i] == modal_norm)
    return Confidence(value=modal_value, score=modal_count / k, method="self_consistency")


def score_response(
    model: Model,
    prompt: str,
    response: ModelResponse,
    *,
    classification: bool,
    true_tokens: frozenset[str] = DEFAULT_TRUE_TOKENS,
    false_tokens: frozenset[str] = DEFAULT_FALSE_TOKENS,
    k: int = 5,
    temperature: float = 0.7,
    base_seed: int = 0,
) -> Confidence:
    """Score a response, preferring logprobs for classification and falling back to
    self-consistency when logprobs are unavailable or the operator is open-ended.
    """
    if classification:
        conf = logprob_confidence(
            response, true_tokens=true_tokens, false_tokens=false_tokens
        )
        if conf is not None:
            return conf
    return self_consistency_confidence(
        model,
        prompt,
        k=k,
        temperature=temperature,
        base_seed=base_seed,
    )
