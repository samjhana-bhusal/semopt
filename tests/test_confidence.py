"""Tests for confidence scoring (FR-5)."""

from __future__ import annotations

import math

from semopt.cascade.confidence import (
    logprob_confidence,
    normalize_answer,
    score_response,
    self_consistency_confidence,
)
from semopt.models.base import ModelResponse
from semopt.models.mock import MockModel


def _resp(logprobs):
    return ModelResponse(
        value="yes", tokens_in=1, tokens_out=1, wall_ms=0.0, logprobs=logprobs
    )


def test_logprob_confidence_picks_max_class():
    # P(yes) via logprob: exp(log(0.8)) = 0.8, P(no) = 0.2
    resp = _resp({"yes": math.log(0.8), "no": math.log(0.2)})
    conf = logprob_confidence(resp)
    assert conf is not None
    assert conf.value == "yes"
    assert abs(conf.score - 0.8) < 1e-6
    assert conf.method == "logprob"


def test_logprob_confidence_renormalizes_over_classes():
    # Non-class tokens present; renormalization over {yes,no} only.
    resp = _resp({"yes": math.log(0.4), "no": math.log(0.1), "maybe": math.log(0.5)})
    conf = logprob_confidence(resp)
    assert conf is not None
    assert conf.value == "yes"
    assert abs(conf.score - 0.8) < 1e-6  # 0.4 / (0.4 + 0.1)


def test_logprob_confidence_none_without_logprobs():
    assert logprob_confidence(_resp(None)) is None


def test_logprob_confidence_none_when_no_class_tokens():
    assert logprob_confidence(_resp({"foo": math.log(0.9)})) is None


def test_normalize_answer():
    assert normalize_answer("  Hello, World! ") == "hello world"
    assert normalize_answer("YES.") == "yes"


def test_self_consistency_all_agree():
    model = MockModel("m", rule=lambda p: "positive", supports_logprobs=False)
    conf = self_consistency_confidence(model, "prompt", k=5)
    assert conf.value == "positive"
    assert conf.score == 1.0
    assert conf.method == "self_consistency"


def test_self_consistency_majority():
    # Answer depends on seed (embedded in prompt by CachingModel normally); here the
    # mock ignores seed, so all samples agree. Use a seed-sensitive rule instead.
    calls = {"i": 0}

    def rule(p: str) -> str:
        calls["i"] += 1
        # 3 of 5 say "a", 2 say "b"
        return "a" if calls["i"] % 5 in (1, 2, 3) else "b"

    model = MockModel("m", rule=rule, supports_logprobs=False)
    conf = self_consistency_confidence(model, "prompt", k=5)
    assert conf.value == "a"
    assert conf.score == 0.6


def test_score_response_falls_back_to_self_consistency():
    # classification requested but no logprobs → self-consistency path.
    model = MockModel("m", rule=lambda p: "yes", supports_logprobs=False)
    resp = ModelResponse(value="yes", tokens_in=1, tokens_out=1, wall_ms=0.0, logprobs=None)
    conf = score_response(model, "prompt", resp, classification=True, k=3)
    assert conf.method == "self_consistency"
    assert conf.value == "yes"
