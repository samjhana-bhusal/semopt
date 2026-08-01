"""Optimizer ablation — the P5 "optimizer breakdown" (SRS §9, Milestone M2).

Attributes the optimizer's estimated cost savings to each pass by turning them on/off:
full optimizer vs. each single pass disabled vs. no optimization at all. Reports the
estimated plan cost (via the cost model + cheap-tier selectivity) for each configuration,
so the contribution of reordering vs. pushdown vs. memoization is visible.

    python experiments/run_ablations.py            # writes outputs/optimizer_ablation.csv

The workload is synthetic (priced API model ids so the cost model produces non-zero
dollars) with (a) filters of differing selectivity and price, and (b) duplicated rows so
memoization has something to save. Real workloads (W1–W3) arrive in P3.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semopt.cost.cost_model import CostModel  # noqa: E402
from semopt.models.mock import KeywordFilterModel  # noqa: E402
from semopt.planner.optimizer import Optimizer, OptimizerConfig, plan_cost  # noqa: E402
from semopt.planner.selectivity import SelectivityEstimator  # noqa: E402
from semopt.table import SemanticTable  # noqa: E402

CONFIGS: dict[str, OptimizerConfig] = {
    "none (naive)": OptimizerConfig(
        reorder=False, pushdown=False, memoize=False, cascade=False
    ),
    "full": OptimizerConfig(cascade=False),
    "-reorder": OptimizerConfig(reorder=False, cascade=False),
    "-pushdown": OptimizerConfig(pushdown=False, cascade=False),
    "-memoize": OptimizerConfig(memoize=False, cascade=False),
}


def make_workload(n_dupe: int = 8) -> SemanticTable:
    """Rows about {alpha (common), beta (rare), gamma (medium)}, each duplicated."""
    freqs = {"alpha": 60, "beta": 8, "gamma": 25}
    rows = []
    for kw, count in freqs.items():
        for i in range(count):
            # Duplicate every row n_dupe times → memoization has repeats to collapse.
            for _ in range(n_dupe):
                rows.append({"text": f"row about {kw} number {i}", "meta": "z"})
    return SemanticTable.from_records(rows)


def build_query(table: SemanticTable):  # type: ignore[no-untyped-def]
    """A deliberately bad plan that each pass improves:

    * a projection and an **expensive map** sit *above* the filters (pushdown moves the
      selective filters in front of them, so the map runs on far fewer rows);
    * two filters of differing price/selectivity (reorder puts the cheap, selective one
      first);
    * duplicated rows (memoize collapses repeats).
    """
    return (
        table.query()
        .project(["text"])
        .map(
            "summarize this row",
            output_column="summary",
            model=KeywordFilterModel("gpt-4o", "alpha"),  # expensive per-row map
            column="text",
        )
        .filter(
            "about alpha?",
            model=KeywordFilterModel("gpt-4o", "alpha"),
            column="text",
            use_cascade=False,
        )
        .filter(
            "about beta?",
            model=KeywordFilterModel("gpt-4o-mini", "beta"),
            column="text",
            use_cascade=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="optimizer ablation (P5)")
    parser.add_argument("--out", default="outputs/optimizer_ablation.csv")
    args = parser.parse_args()

    cost_model = CostModel.from_yaml()
    table = make_workload()
    base_query = build_query(table)
    # One estimator (shared sample) so selectivity/unique-fraction are consistent.
    estimator = SelectivityEstimator(table, sample_size=1000)

    rows = []
    naive_cost = None
    for name, cfg in CONFIGS.items():
        optimized = Optimizer(cfg).optimize(base_query)
        cost = plan_cost(optimized, cost_model, estimator).total_usd
        if name.startswith("none"):
            naive_cost = cost
        rows.append({"config": name, "estimated_cost_usd": cost})

    for r in rows:
        r["fraction_of_naive"] = (
            r["estimated_cost_usd"] / naive_cost if naive_cost else 0.0
        )
        r["savings_vs_naive_pct"] = 100.0 * (1.0 - r["fraction_of_naive"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"optimizer ablation (P5) — {len(table)} rows\n")
    header = f"{'config':>14} {'est_cost_$':>12} {'vs_naive':>9} {'savings':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['config']:>14} {r['estimated_cost_usd']:>12.6f} "
            f"{r['fraction_of_naive']:>8.2f}x {r['savings_vs_naive_pct']:>8.1f}%"
        )
    print(f"\nwrote {out}")
    print(
        "\nRead: disabling one pass (rows starting '-') shows that pass's contribution — "
        "the cost rises back toward naive by roughly what that pass was saving."
    )


if __name__ == "__main__":
    main()
