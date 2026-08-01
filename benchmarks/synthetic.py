"""Synthetic labeled workload for coverage validation (P1 / Milestone M1).

Generates a binary ``sem_filter``-style workload with a *known* difficulty structure so
the conformal cascade's coverage guarantee can be checked against ground truth before any
real benchmark is wired up (SRS W6). Each row has:

* a latent difficulty ``d ∈ [0, 1]``;
* a ground-truth boolean label;
* a **cheap** model that is confident and usually right on easy rows, and unconfident and
  often wrong on hard rows — so its confidence is genuinely informative;
* an **expensive** model that is accurate regardless of difficulty.

The cheap/expensive behaviors are materialized as :class:`LookupModel` lookup tables keyed
by the exact prompt string, making the whole workload deterministic (NFR-2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semopt.models.mock import LookupModel


@dataclass(frozen=True)
class SyntheticWorkload:
    prompts: list[str]
    labels: list[bool]
    cheap: LookupModel
    expensive: LookupModel

    def __len__(self) -> int:
        return len(self.prompts)


@dataclass(frozen=True)
class SplitWorkload:
    """One shared cheap/expensive model pair over a cal/test split of the same rows.

    This mirrors real usage — a single model answers any prompt — where the synthetic
    :class:`LookupModel` only knows the prompts it was built with. Baselines calibrate on
    the cal split and evaluate on the test split using the *same* models.
    """

    cheap: LookupModel
    expensive: LookupModel
    cal_prompts: list[str]
    cal_labels: list[bool]
    test_prompts: list[str]
    test_labels: list[bool]


def _val(label: bool) -> str:
    return "yes" if label else "no"


def _wrong(label: bool) -> str:
    return "no" if label else "yes"


def _generate(
    n: int, seed: int, expensive_accuracy: float, prefix: str
) -> tuple[list[str], list[bool], dict[str, tuple[str, float]], dict[str, tuple[str, float]]]:
    rng = np.random.default_rng(seed)
    prompts: list[str] = []
    labels: list[bool] = []
    cheap_table: dict[str, tuple[str, float]] = {}
    exp_table: dict[str, tuple[str, float]] = {}

    for i in range(n):
        p = f"{prefix}-{seed}-{i}"
        d = float(rng.uniform())
        label = bool(rng.integers(0, 2))

        # Cheap tier: accuracy falls from ~1.0 (easy) to ~0.5 (hard).
        p_cheap_correct = 0.5 + 0.5 * (1.0 - d)
        cheap_correct = rng.uniform() < p_cheap_correct
        cheap_val = _val(label) if cheap_correct else _wrong(label)
        # Confidence high on easy rows, lower on hard, with noise; clipped to (0.5, 1).
        cheap_conf = float(np.clip(0.55 + 0.45 * (1.0 - d) + rng.normal(0, 0.05), 0.5, 0.999))

        # Expensive tier: accurate regardless of difficulty.
        exp_correct = rng.uniform() < expensive_accuracy
        exp_val = _val(label) if exp_correct else _wrong(label)

        prompts.append(p)
        labels.append(label)
        cheap_table[p] = (cheap_val, cheap_conf)
        exp_table[p] = (exp_val, 0.9)
    return prompts, labels, cheap_table, exp_table


def make_workload(
    n: int,
    *,
    seed: int = 42,
    expensive_accuracy: float = 0.97,
    prefix: str = "row",
) -> SyntheticWorkload:
    """Generate ``n`` rows with the difficulty structure described above."""
    prompts, labels, cheap_table, exp_table = _generate(n, seed, expensive_accuracy, prefix)
    return SyntheticWorkload(
        prompts=prompts,
        labels=labels,
        cheap=LookupModel("mlx-llama-3.2-1b", cheap_table),
        expensive=LookupModel("gpt-4o", exp_table),
    )


def make_split_workload(
    n: int,
    *,
    seed: int = 42,
    cal_frac: float = 0.5,
    expensive_accuracy: float = 0.97,
) -> SplitWorkload:
    """Generate ``n`` rows and split into cal/test, sharing one cheap/expensive pair."""
    prompts, labels, cheap_table, exp_table = _generate(n, seed, expensive_accuracy, "split")
    n_cal = int(n * cal_frac)
    cheap = LookupModel("mlx-llama-3.2-1b", cheap_table)
    expensive = LookupModel("gpt-4o", exp_table)
    return SplitWorkload(
        cheap=cheap,
        expensive=expensive,
        cal_prompts=prompts[:n_cal],
        cal_labels=labels[:n_cal],
        test_prompts=prompts[n_cal:],
        test_labels=labels[n_cal:],
    )
