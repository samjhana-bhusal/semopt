"""MLX local-model backend (FR-3.2).

Runs Llama-3.2-1B / 3.2-3B / 3.1-8B locally on Apple silicon via ``mlx-lm``, exposing
token-level logprobs needed for confidence scoring (FR-5.1). ``mlx``/``mlx-lm`` are
optional extras; importing this module without them raises a clear error only when a
backend is actually constructed, so the core engine imports cleanly everywhere.

Status: interface + generation path scaffolded. The logprob-extraction path is marked
with a TODO to verify against the installed ``mlx-lm`` API on the target M5 (SRS §14 Q1,
Risk R1) — spike before trusting cascade confidence.
"""

from __future__ import annotations

import time

from semopt.models.base import Model, ModelResponse

_MODEL_IDS = {
    "mlx-llama-3.2-1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-llama-3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-llama-3.1-8b": "mlx-community/Llama-3.1-8B-Instruct-4bit",
}


class MLXModel(Model):
    supports_logprobs = True

    def __init__(self, model_id: str, *, max_tokens: int = 256) -> None:
        if model_id not in _MODEL_IDS:
            raise ValueError(
                f"unknown MLX model {model_id!r}; known: {sorted(_MODEL_IDS)}"
            )
        try:
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - requires optional dep
            raise ImportError(
                "MLX backend requires the 'mlx' extra: pip install -e '.[mlx]'"
            ) from exc

        self.model_id = model_id
        self._hf_repo = _MODEL_IDS[model_id]
        self._default_max_tokens = max_tokens
        self._model, self._tokenizer = load(self._hf_repo)

    def predict(
        self,
        prompt: str,
        *,
        examples: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> ModelResponse:  # pragma: no cover - requires optional dep + weights
        from mlx_lm import generate

        full_prompt = self._format(prompt, examples)
        t0 = time.perf_counter()
        text = generate(
            self._model,
            self._tokenizer,
            prompt=full_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0

        tokens_in = len(self._tokenizer.encode(full_prompt))
        tokens_out = len(self._tokenizer.encode(text))
        # TODO(spike, R1): extract per-token logprobs from mlx_lm's generate_step to
        # populate ModelResponse.logprobs. Until verified on-device, leave None so the
        # confidence layer uses self-consistency (FR-5.2) rather than trusting a guess.
        return ModelResponse(
            value=text.strip(),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            wall_ms=wall_ms,
            logprobs=None,
            model_id=self.model_id,
        )

    def _format(self, prompt: str, examples: list[tuple[str, str]] | None) -> str:
        messages = []
        for inp, out in examples or []:
            messages.append({"role": "user", "content": inp})
            messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": prompt})
        if hasattr(self._tokenizer, "apply_chat_template"):
            return str(
                self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            )
        return prompt
