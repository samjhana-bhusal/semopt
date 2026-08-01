"""Emit the per-row trace (FR-10.2) and cascade dashboard (FR-10.3) for one run.

Calibrates a 2-tier conformal cascade on the synthetic split, runs it on the test set,
and writes:

    outputs/trace.jsonl            — one JSON record per row (audit trail)
    outputs/cascade_dashboard.html — tier distribution, cost, coverage vs. α

    python experiments/run_dashboard.py [--alpha 0.10] [--n 2000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.synthetic import make_split_workload  # noqa: E402

from semopt.cascade.cascade import Cascade, Tier  # noqa: E402
from semopt.eval.baselines import conformal_two_tier_tau, default_is_correct  # noqa: E402
from semopt.report.dashboard import write_dashboard  # noqa: E402
from semopt.report.trace import write_trace  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="emit trace + dashboard (FR-10.2/10.3)")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()

    wl = make_split_workload(args.n, seed=42)
    tau, _ = conformal_two_tier_tau(
        wl.cheap, wl.expensive, wl.cal_prompts, wl.cal_labels, alpha=args.alpha
    )
    cascade = Cascade([
        Tier(wl.cheap, classification=True, tau=tau),
        Tier(wl.expensive, classification=True, tau=float("-inf")),
    ])
    results = cascade.run(wl.test_prompts)
    correct = [default_is_correct(r.value, y) for r, y in zip(results, wl.test_labels, strict=True)]

    n = write_trace(results, "outputs/trace.jsonl")
    stats = write_dashboard(
        results,
        "outputs/cascade_dashboard.html",
        title=f"Conformal cascade · synthetic · α={args.alpha:.2f}",
        correct=correct,
        alpha=args.alpha,
    )

    print(f"wrote outputs/trace.jsonl ({n} rows)")
    print("wrote outputs/cascade_dashboard.html")
    print(
        f"  tier0 served {stats.tier_fractions.get(0, 0) * 100:.1f}% | "
        f"coverage {stats.empirical_coverage:.3f} (target {1 - args.alpha:.2f}) | "
        f"total ${stats.total_cost_usd:.6f}"
    )


if __name__ == "__main__":
    main()
