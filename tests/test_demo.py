"""Smoke test for the M0 demo (SRS Milestone M0)."""

from __future__ import annotations

from semopt.cache.memo import CachingModel, LLMCache
from semopt.cost.cost_model import CostModel
from semopt.demo import make_reviews, resolve_cheap_model, resolve_expensive_model


def test_make_reviews_deterministic():
    a = make_reviews(100)
    b = make_reviews(100)
    assert len(a) == 100
    assert list(a["text"]) == list(b["text"])  # fixed seed → identical


def test_demo_offline_end_to_end(tmp_path):
    reviews = make_reviews(100)
    cache = LLMCache(tmp_path / "demo.sqlite")
    cost_model = CostModel.from_yaml()

    cheap = CachingModel(resolve_cheap_model(real=False), cache)
    out_cheap = reviews.sem_filter(
        "Does this review complain about shipping or packaging?", model=cheap
    )
    # Some but not all rows kept — the filter actually discriminates.
    assert 0 < len(out_cheap) < 100
    assert cheap.misses == 100 and cheap.hits == 0

    # Second pass over the same cache: pure replay, no backend calls (NFR-2).
    cheap2 = CachingModel(resolve_cheap_model(real=False), cache)
    reviews.sem_filter(
        "Does this review complain about shipping or packaging?", model=cheap2
    )
    assert cheap2.hits == 100 and cheap2.misses == 0

    # Expensive tier is API-priced; cost must be strictly positive.
    expensive = CachingModel(resolve_expensive_model(real=False), cache)
    out_exp = reviews.sem_filter(
        "Does this review complain about shipping or packaging?", model=expensive
    )
    prov = out_exp.provenance(out_exp.columns[0])
    assert cost_model.get("gpt-4o").tier == "expensive"
    assert prov.cost_usd > 0.0
