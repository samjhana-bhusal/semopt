"""Semantic memoization / deterministic LLM replay (FR-3.4, FR-9).

A persistent SQLite store keyed by ``(model_id, canonicalized_prompt_hash)``. Wrapping
any :class:`~semopt.models.base.Model` in :class:`CachingModel` makes repeated or
replayed runs hit the cache instead of the backend, which is what gives the eval
pipeline determinism (NFR-2) and cost containment (NFR-4).

Prompt canonicalization (FR-9.1): strip surrounding whitespace and collapse internal
runs of whitespace so cosmetically different prompts share a cache entry. Structured
(JSON-object) prompts additionally get their keys sorted.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path

from semopt.models.base import Model, ModelResponse

_WS = re.compile(r"\s+")


def canonicalize_prompt(prompt: str) -> str:
    """Normalize a prompt for cache keying (FR-9.1)."""
    stripped = prompt.strip()
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return _WS.sub(" ", stripped)
    if isinstance(obj, dict):
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return _WS.sub(" ", stripped)


def prompt_hash(model_id: str, prompt: str) -> str:
    canon = canonicalize_prompt(prompt)
    return hashlib.sha256(f"{model_id}\x00{canon}".encode()).hexdigest()


class LLMCache:
    """SQLite-backed key/value store for :class:`ModelResponse` objects."""

    def __init__(self, path: str | Path = "outputs/llm_cache.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key         TEXT PRIMARY KEY,
                model_id    TEXT NOT NULL,
                value       TEXT NOT NULL,
                tokens_in   INTEGER NOT NULL,
                tokens_out  INTEGER NOT NULL,
                wall_ms     REAL NOT NULL,
                logprobs    TEXT
            )
            """
        )
        self._conn.commit()

    def get(self, model_id: str, prompt: str) -> ModelResponse | None:
        key = prompt_hash(model_id, prompt)
        with self._lock:
            row = self._conn.execute(
                "SELECT value, tokens_in, tokens_out, wall_ms, logprobs "
                "FROM llm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        value, tokens_in, tokens_out, wall_ms, logprobs_json = row
        logprobs = json.loads(logprobs_json) if logprobs_json is not None else None
        return ModelResponse(
            value=value,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            wall_ms=wall_ms,
            logprobs=logprobs,
            model_id=model_id,
        )

    def put(self, model_id: str, prompt: str, response: ModelResponse) -> None:
        key = prompt_hash(model_id, prompt)
        logprobs_json = (
            json.dumps(response.logprobs) if response.logprobs is not None else None
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache "
                "(key, model_id, value, tokens_in, tokens_out, wall_ms, logprobs) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    model_id,
                    response.value,
                    response.tokens_in,
                    response.tokens_out,
                    response.wall_ms,
                    logprobs_json,
                ),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])

    def close(self) -> None:
        self._conn.close()


class CachingModel(Model):
    """Wrap a backend so calls are served from / written to an :class:`LLMCache`.

    ``read_only`` (``--dry-run`` in experiments, NFR-4) forbids backend calls entirely:
    a cache miss raises rather than spending money or compute.
    """

    def __init__(
        self,
        backend: Model,
        cache: LLMCache,
        *,
        read_only: bool = False,
    ) -> None:
        self._backend = backend
        self._cache = cache
        self.read_only = read_only
        self.model_id = backend.model_id
        self.supports_logprobs = backend.supports_logprobs
        self.hits = 0
        self.misses = 0

    def predict(
        self,
        prompt: str,
        *,
        examples: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> ModelResponse:
        # Few-shot examples and sampling params are part of the effective prompt: fold
        # them into the cache key so different call configs don't collide.
        cache_prompt = self._effective_prompt(prompt, examples, temperature, max_tokens, seed)
        cached = self._cache.get(self.model_id, cache_prompt)
        if cached is not None:
            self.hits += 1
            return cached
        if self.read_only:
            raise RuntimeError(
                f"cache miss for model={self.model_id!r} in read_only mode "
                f"(dry-run): refusing to call backend"
            )
        self.misses += 1
        response = self._backend.predict(
            prompt,
            examples=examples,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        self._cache.put(self.model_id, cache_prompt, response)
        return response

    @staticmethod
    def _effective_prompt(
        prompt: str,
        examples: list[tuple[str, str]] | None,
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> str:
        payload = {
            "prompt": prompt,
            "examples": examples or [],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
