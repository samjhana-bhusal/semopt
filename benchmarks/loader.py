"""Common workload interface for real benchmarks (SRS §7).

Every real workload (W1–W3) resolves to the same shape the eval harness already consumes
from the synthetic generator: a shared model pair plus cal/test prompt+label splits. Once
a loader returns a :class:`LabeledSplit`, the baselines, coverage, and Pareto experiments
run against it unchanged.

The loaders here are **stubs**: they document the source/license/preprocessing and define
the split contract, but do not download data (that needs network + a real model backend
and is gated on the applicant's API budget, SRS NFR-4). Each raises with the exact steps
to wire it up. Fixed 80/10/10 train/cal/test splits with seed 42 (SRS §7 data hygiene).
"""

from __future__ import annotations

from dataclasses import dataclass

from semopt.models.base import Model


@dataclass(frozen=True)
class LabeledSplit:
    """A workload ready for the eval harness."""

    name: str
    cheap: Model
    expensive: Model
    cal_prompts: list[str]
    cal_labels: list[object]
    test_prompts: list[str]
    test_labels: list[object]
    task: str  # "filter" | "extract" | "rank"


SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)  # train / cal / test


def _not_wired(name: str, steps: str) -> LabeledSplit:
    raise NotImplementedError(
        f"benchmark {name!r} is not wired up in this environment.\n"
        f"To enable it:\n{steps}\n"
        f"It must return a LabeledSplit; the eval harness then runs unchanged."
    )


def load_product_reviews() -> LabeledSplit:
    """W1 — Amazon Reviews 2023 (McAuley Lab), 10K Electronics; sem_filter (SRS §7 W1)."""
    return _not_wired(
        "product_reviews",
        "  1. Download the Electronics 5-core subset from the McAuley Lab release.\n"
        "  2. Sample 10K reviews (seed 42); hand-label 500 for the test set.\n"
        "  3. Prompt = 'Does this review complain about shipping or packaging?'\n"
        "  4. Build cheap+expensive Model backends (MLX + API) and 80/10/10 splits.",
    )


def load_arxiv_abstracts() -> LabeledSplit:
    """W2 — arXiv cs.LG abstracts (Kaggle mirror), 10K; filter+extract+rank (SRS §7 W2)."""
    return _not_wired(
        "arxiv_abstracts",
        "  1. Pull the arXiv metadata dump; take 10K cs.LG abstracts.\n"
        "  2. Manually label a 200-row held-out set for 'about learned indexes'.\n"
        "  3. Tasks: sem_filter, sem_extract{contribution_type, evaluation_dataset},\n"
        "     sem_rank top-20 by 'most likely a systems contribution'.",
    )


def load_contract_review() -> LabeledSplit:
    """W3 — CUAD (Contract Understanding Atticus Dataset), 510 contracts (SRS §7 W3)."""
    return _not_wired(
        "contract_review",
        "  1. Download CUAD v1 (expert-labeled clauses; labels already provided).\n"
        "  2. Task: sem_extract clause type + risk category per paragraph.\n"
        "  3. Use CUAD's own labels as ground truth; 80/10/10 split, seed 42.",
    )


LOADERS = {
    "product_reviews": load_product_reviews,
    "arxiv_abstracts": load_arxiv_abstracts,
    "contract_review": load_contract_review,
}
