"""Cost–accuracy Pareto experiment (SRS §9 plot P1, Milestone M3).

Runs every baseline (B0–B3, B5) and our conformal cascade on a shared cal/test split,
sweeping α for our method to trace a cost–accuracy frontier. Writes ``outputs/pareto.csv``
with one row per (method, α) point: cost, accuracy, F1, and escalation fraction.

    python experiments/run_pareto.py            # writes outputs/pareto.csv
    python experiments/run_pareto.py --n 4000

On the synthetic workload the expected ordering is: naive-strongest is most accurate and
most expensive; naive-cheapest is cheapest and least accurate; our conformal cascade sits
on the frontier, matching the strong model's accuracy target at a fraction of its cost,
and dominating the fixed-threshold and hand-tuned baselines (which lack the guarantee).
Real workloads (W1–W3) plug into the same harness once their loaders return prompts+labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.synthetic import make_split_workload  # noqa: E402

from semopt.cost.cost_model import CostModel  # noqa: E402
from semopt.eval.baselines import run_all_baselines  # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)


def main() -> None:
    parser = argparse.ArgumentParser(description="cost-accuracy Pareto (P1)")
    parser.add_argument("--n", type=int, default=3000, help="total rows (split cal/test)")
    parser.add_argument("--out", default="outputs/pareto.csv")
    args = parser.parse_args()

    wl = make_split_workload(args.n, seed=42)
    cost_model = CostModel.from_yaml()
    n_test = len(wl.test_prompts)

    rows: list[dict[str, object]] = []
    seen_alpha_independent = False
    for alpha in ALPHAS:
        results = run_all_baselines(
            wl.cheap, wl.expensive,
            wl.cal_prompts, wl.cal_labels,
            wl.test_prompts, wl.test_labels,
            alpha=alpha, cost_model=cost_model,
        )
        alpha_free = {"B0_naive_strongest", "B1_naive_cheapest", "B3_fixed_threshold"}
        for r in results:
            # B0/B1/B3 don't depend on α — emit them once (at the first α).
            alpha_independent = r.name in alpha_free
            if alpha_independent and seen_alpha_independent:
                continue
            rows.append({
                "method": r.name,
                "alpha": "" if alpha_independent else alpha,
                "accuracy": round(r.accuracy, 4),
                "f1": round(r.f1, 4),
                "cost_usd": round(r.cost_usd, 6),
                "cost_per_1k": round(r.cost_usd / n_test * 1000, 6),
                "escalate_frac": round(r.escalate_frac, 4),
                "tau": "" if r.tau is None else round(r.tau, 4),
            })
        seen_alpha_independent = True

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"cost-accuracy Pareto (n_test={n_test})\n")
    header = f"{'method':>24} {'alpha':>6} {'acc':>6} {'f1':>6} {'cost/1k$':>9} {'escal':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        a = f"{r['alpha']:.2f}" if r["alpha"] != "" else "  -"
        print(
            f"{r['method']:>24} {a:>6} {r['accuracy']:>6.3f} {r['f1']:>6.3f} "
            f"{r['cost_per_1k']:>9.4f} {r['escalate_frac']:>6.2f}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
