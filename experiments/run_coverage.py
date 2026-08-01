"""Coverage validation experiment (SRS W6, Milestone M1; eval P2 / SC2).

Demonstrates that the conformal cascade holds its accuracy target: across
α ∈ {0.05, 0.10, 0.20}, empirical end-to-end coverage on a held-out test set lands within
±3% of the nominal 1−α (SC2), while costing a fraction of the naive-strongest baseline.

Budgeting (the rigorous 2-tier case): end-to-end error ≈ leaked_0 + escalate_frac·err_strong,
where leaked_0 = P(cheap wrong ∧ accepted) and err_strong is the strong tier's own error,
both measured on the calibration set. We calibrate the cheap tier's leaked-error budget to
α₀ = α − escalate_frac·err_strong; since escalate_frac itself depends on α₀, we solve the
fixed point (a contraction). When err_strong ≥ α the target is *unattainable* even with the
strongest model — a genuine finding (SRS Risk R3), reported rather than hidden.

    python experiments/run_coverage.py                 # writes outputs/coverage_sweep.csv
    python experiments/run_coverage.py --n 4000
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script (python experiments/run_coverage.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.synthetic import make_workload  # noqa: E402

from semopt.cascade.cascade import Cascade, Tier  # noqa: E402
from semopt.cost.cost_model import CostModel  # noqa: E402
from semopt.eval.baselines import conformal_two_tier_tau  # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)


def _is_correct(value: str, label: object) -> bool:
    return (value == "yes") == bool(label)


def _bootstrap_ci(
    correct: np.ndarray, n_resamples: int = 100, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap CI for the coverage (mean correctness) (Eval Metric 3)."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = [correct[rng.integers(0, n, n)].mean() for _ in range(n_resamples)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def run_one(alpha: float, n: int, cost_model: CostModel) -> dict[str, float]:
    cal = make_workload(n, seed=42, prefix="cal")
    test = make_workload(n, seed=7, prefix="test")

    # Calibrate the cheap tier's τ via the shared 2-tier conformal helper (fixed-point
    # budget α₀ = α − escalate_frac · err_strong; see semopt.eval.baselines).
    tau, err_strong = conformal_two_tier_tau(
        cal.cheap, cal.expensive, cal.prompts, cal.labels,
        alpha=alpha, is_correct=_is_correct,
    )
    alpha0_eff = max(alpha - err_strong, 1e-3)
    attainable = (alpha - err_strong) > 0.0

    # Rebuild the cascade over the *test* models but reuse the calibrated threshold.
    test_tiers = [
        Tier(test.cheap, classification=True, tau=tau),
        Tier(test.expensive, classification=True, tau=float("-inf")),
    ]
    cascade = Cascade(test_tiers, cost_model=cost_model)
    results = cascade.run(test.prompts)

    correct = np.array(
        [_is_correct(r.value, y) for r, y in zip(results, test.labels, strict=True)]
    )
    coverage = float(correct.mean())
    ci_lo, ci_hi = _bootstrap_ci(correct)

    dist = cascade.tier_distribution(results)
    escalate_frac = float(sum(v for k, v in dist.items() if k > 0))

    cost_cascade = float(sum(r.total_cost_usd for r in results))
    strong_cost_per = cost_model.call_cost("gpt-4o", 12, 1)
    cost_naive = strong_cost_per * len(test)
    cost_fraction = cost_cascade / cost_naive if cost_naive > 0 else 0.0

    return {
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "empirical_coverage": coverage,
        "coverage_ci_lo": ci_lo,
        "coverage_ci_hi": ci_hi,
        "coverage_gap": coverage - (1 - alpha),
        "within_3pct": float(abs(coverage - (1 - alpha)) <= 0.03),
        "err_strong": err_strong,
        "alpha0_budget": alpha0_eff,
        "attainable": float(attainable),
        "tier0_accept_rate": float(dist.get(0, 0.0)),
        "escalate_frac": escalate_frac,
        "cost_cascade_usd": cost_cascade,
        "cost_naive_usd": cost_naive,
        "cost_fraction": cost_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="conformal cascade coverage sweep")
    parser.add_argument("--n", type=int, default=2000, help="rows per split")
    parser.add_argument("--out", default="outputs/coverage_sweep.csv")
    args = parser.parse_args()

    cost_model = CostModel.from_yaml()
    rows = [run_one(alpha, args.n, cost_model) for alpha in ALPHAS]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"conformal cascade coverage sweep (n={args.n} per split)\n")
    header = (
        f"{'alpha':>6} {'nominal':>8} {'empirical':>10} {'gap':>7} "
        f"{'±3%':>4} {'escal':>7} {'cost_frac':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['alpha']:>6.2f} {r['nominal_coverage']:>8.2f} "
            f"{r['empirical_coverage']:>10.3f} {r['coverage_gap']:>+7.3f} "
            f"{'yes' if r['within_3pct'] else 'NO':>4} "
            f"{r['escalate_frac']:>7.2f} {r['cost_fraction']:>10.3f}"
        )
    print(f"\nwrote {out}")
    print(
        "\nRead: 'empirical' should hug 'nominal' (SC2, ±3%); 'cost_frac' is the fraction "
        "of naive-strongest cost paid; 'escal' is the fraction escalated to the strong tier."
    )


if __name__ == "__main__":
    main()
