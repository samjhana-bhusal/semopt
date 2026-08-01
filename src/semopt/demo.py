"""M0 end-to-end demo (SRS §10, Milestone M0).

Runs ``sem_filter`` over 100 rows using a cheap (local/MLX) tier and an expensive
(API) tier, wrapped in the persistent LLM cache, and prints the logged cost for each.

Runs fully offline by default: if MLX weights or API keys are unavailable, the demo
substitutes deterministic mock backends that *impersonate* the real model ids, so the
cost table still applies and the output shape is identical. Pass ``--real`` to require
the real backends (and fail loudly if they are missing).

    python -m semopt.demo             # offline, deterministic
    python -m semopt.demo --rows 200  # more rows
    python -m semopt.demo --real      # require MLX + API backends
"""

from __future__ import annotations

import argparse
import random

from semopt.cache.memo import CachingModel, LLMCache
from semopt.cost.cost_model import CostModel
from semopt.models.base import Model
from semopt.models.mock import KeywordFilterModel
from semopt.table import SemanticTable

_COMPLAINT_TEMPLATES = [
    "The shipping was late and the box arrived crushed.",
    "Packaging was destroyed, item was damaged in transit.",
    "Delivery took three weeks, totally unacceptable shipping.",
    "Great product, works exactly as described.",
    "Love it, five stars, would buy again.",
    "Battery life is excellent and setup was easy.",
    "Item never shipped, still waiting after a month.",
    "Perfect fit, high quality, no complaints at all.",
]


def make_reviews(n: int, seed: int = 42) -> SemanticTable:
    rng = random.Random(seed)
    records = [
        {"id": i, "text": rng.choice(_COMPLAINT_TEMPLATES)} for i in range(n)
    ]
    return SemanticTable.from_records(records)


def resolve_cheap_model(real: bool) -> Model:
    if real:
        from semopt.models.mlx_backend import MLXModel

        return MLXModel("mlx-llama-3.2-1b")
    # Offline stand-in impersonating the cheap MLX tier (free in the cost table).
    return KeywordFilterModel("mlx-llama-3.2-1b", keyword="shipping")


def resolve_expensive_model(real: bool) -> Model:
    if real:
        from semopt.models.api_backend import OpenAIModel

        return OpenAIModel("gpt-4o")
    # Offline stand-in impersonating the expensive API tier (priced in the cost table).
    return KeywordFilterModel("gpt-4o", keyword="shipping")


def run_tier(
    label: str,
    backend: Model,
    reviews: SemanticTable,
    cache: LLMCache,
    cost_model: CostModel,
) -> None:
    model = CachingModel(backend, cache)
    out = reviews.sem_filter(
        "Does this review complain about shipping or packaging?",
        model=model,
    )
    prov = out.provenance(out.columns[0])
    mc = cost_model.get(backend.model_id)
    print(f"\n[{label}] model={backend.model_id}  (tier={mc.tier})")
    print(f"  rows in={len(reviews)}  kept={len(out)}")
    print(f"  cache: {model.hits} hits / {model.misses} misses")
    print(f"  tokens: in={prov.tokens_in}  out={prov.tokens_out}")
    print(f"  cost:  ${prov.cost_usd:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="semopt M0 demo")
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--real", action="store_true", help="require MLX + API backends")
    parser.add_argument("--cache", default="outputs/demo_cache.sqlite")
    args = parser.parse_args()

    reviews = make_reviews(args.rows)
    cache = LLMCache(args.cache)
    cost_model = CostModel.from_yaml()

    mode = "REAL backends" if args.real else "offline (mock backends)"
    print(f"semopt M0 demo — {mode}")
    print(f"filtering {len(reviews)} reviews for shipping/packaging complaints")

    run_tier("cheap", resolve_cheap_model(args.real), reviews, cache, cost_model)
    run_tier("expensive", resolve_expensive_model(args.real), reviews, cache, cost_model)

    print("\nNote: cheap tier is a local model (~$0); the expensive tier is API-priced.")
    print("This is the naive single-model path — the conformal cascade lands in P1.")
    cache.close()


if __name__ == "__main__":
    main()
