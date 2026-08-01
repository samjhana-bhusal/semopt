"""API model backends for OpenAI and Anthropic (FR-3.3).

Optional extras (``.[api]``). Both are imported lazily so the core engine runs without
the SDKs or any key. OpenAI can return token logprobs; Anthropic does not, so it reports
``supports_logprobs = False`` and the confidence layer falls back to self-consistency
(FR-5.2). Keys are read from the environment (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``);
they are never logged.
"""

from __future__ import annotations

import os
import time

from semopt.models.base import Model, ModelResponse

_OPENAI_MODELS = {"gpt-4o-mini": "gpt-4o-mini", "gpt-4o": "gpt-4o"}
_ANTHROPIC_MODELS = {
    "claude-haiku": "claude-haiku-4-5-20251001",
    "claude-sonnet": "claude-sonnet-4-5",
}


class OpenAIModel(Model):
    supports_logprobs = True

    def __init__(self, model_id: str) -> None:
        if model_id not in _OPENAI_MODELS:
            raise ValueError(f"unknown OpenAI model {model_id!r}")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("OpenAI backend requires: pip install -e '.[api]'") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        self.model_id = model_id
        self._api_name = _OPENAI_MODELS[model_id]
        self._client = OpenAI()

    def predict(
        self,
        prompt: str,
        *,
        examples: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> ModelResponse:  # pragma: no cover - requires network + key
        messages = []
        for inp, out in examples or []:
            messages.append({"role": "user", "content": inp})
            messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self._api_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=True,
            top_logprobs=5,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        choice = resp.choices[0]
        value = (choice.message.content or "").strip()

        logprobs: dict[str, float] | None = None
        if choice.logprobs and choice.logprobs.content:
            top = choice.logprobs.content[0]
            logprobs = {tl.token: tl.logprob for tl in top.top_logprobs}

        usage = resp.usage
        return ModelResponse(
            value=value,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            wall_ms=wall_ms,
            logprobs=logprobs,
            model_id=self.model_id,
        )


class AnthropicModel(Model):
    supports_logprobs = False  # Anthropic API does not expose token logprobs (FR-3.3).

    def __init__(self, model_id: str) -> None:
        if model_id not in _ANTHROPIC_MODELS:
            raise ValueError(f"unknown Anthropic model {model_id!r}")
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("Anthropic backend requires: pip install -e '.[api]'") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.model_id = model_id
        self._api_name = _ANTHROPIC_MODELS[model_id]
        self._client = Anthropic()

    def predict(
        self,
        prompt: str,
        *,
        examples: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> ModelResponse:  # pragma: no cover - requires network + key
        messages = []
        for inp, out in examples or []:
            messages.append({"role": "user", "content": inp})
            messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=self._api_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        value = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
        return ModelResponse(
            value=value,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            wall_ms=wall_ms,
            logprobs=None,
            model_id=self.model_id,
        )
