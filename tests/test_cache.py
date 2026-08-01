"""Tests for the LLM cache / deterministic replay (FR-3.4, FR-9)."""

from __future__ import annotations

import pytest

from semopt.cache.memo import CachingModel, LLMCache, canonicalize_prompt, prompt_hash
from semopt.models.mock import MockModel


def test_canonicalize_collapses_whitespace():
    assert canonicalize_prompt("  hello   world  ") == "hello world"
    assert canonicalize_prompt("a\n\nb") == "a b"


def test_canonicalize_sorts_json_keys():
    a = canonicalize_prompt('{"b": 1, "a": 2}')
    b = canonicalize_prompt('{"a": 2, "b": 1}')
    assert a == b


def test_prompt_hash_stable_across_cosmetic_diffs():
    assert prompt_hash("m", "hello   world") == prompt_hash("m", "  hello world  ")
    assert prompt_hash("m1", "x") != prompt_hash("m2", "x")


def test_cache_roundtrip(tmp_path):
    cache = LLMCache(tmp_path / "c.sqlite")
    calls = {"n": 0}

    def rule(p: str) -> str:
        calls["n"] += 1
        return "answer"

    backend = MockModel("mock-cheap", rule=rule)
    cached = CachingModel(backend, cache)

    r1 = cached.predict("what is this?")
    r2 = cached.predict("what is this?")
    assert r1.value == r2.value == "answer"
    assert calls["n"] == 1  # backend hit only once
    assert cached.hits == 1
    assert cached.misses == 1
    assert len(cache) == 1


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.sqlite"
    backend = MockModel("mock-cheap", rule=lambda p: "v")
    CachingModel(backend, LLMCache(path)).predict("x")

    # New cache instance over the same file must serve the entry.
    misses = {"n": 0}

    def rule(p: str) -> str:
        misses["n"] += 1
        return "v"

    cached2 = CachingModel(MockModel("mock-cheap", rule=rule), LLMCache(path))
    cached2.predict("x")
    assert misses["n"] == 0
    assert cached2.hits == 1


def test_read_only_raises_on_miss(tmp_path):
    cache = LLMCache(tmp_path / "c.sqlite")
    backend = MockModel("mock-cheap", rule=lambda p: "v")
    cached = CachingModel(backend, cache, read_only=True)
    with pytest.raises(RuntimeError, match="read_only"):
        cached.predict("uncached prompt")
