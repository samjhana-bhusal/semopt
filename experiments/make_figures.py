"""Render paper figures from the experiment CSVs (SRS §9 plots P1, P2, P5).

Reads the CSVs produced by ``run_pareto.py`` / ``run_coverage.py`` / ``run_ablations.py``
and writes PNGs under ``report/figures/``. Kept separate from the experiments so plotting
never blocks a run and figures are always traceable to a specific CSV (NFR-1).

    python experiments/make_figures.py     # needs the .[plots] extra (matplotlib)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, no display
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "report" / "figures"


def _read(path: Path) -> list[dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(row: dict[str, str], key: str) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def figure_pareto() -> None:
    """P1 — cost (x) vs accuracy (y), our frontier + baselines."""
    rows = _read(OUT / "pareto.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    ours = [r for r in rows if r["method"] == "ours_conformal"]
    ours.sort(key=lambda r: _f(r, "cost_per_1k"))
    ax.plot(
        [_f(r, "cost_per_1k") for r in ours],
        [_f(r, "accuracy") for r in ours],
        "-o", color="#2b6cb0", linewidth=2, markersize=7,
        label="ours (conformal cascade)", zorder=3,
    )
    for r in ours:
        ax.annotate(f"α={_f(r, 'alpha'):.2f}",
                    (_f(r, "cost_per_1k"), _f(r, "accuracy")),
                    textcoords="offset points", xytext=(6, -10), fontsize=8, color="#2b6cb0")

    markers = {
        "B0_naive_strongest": ("naive strongest", "#e53e3e", "^"),
        "B1_naive_cheapest": ("naive cheapest", "#718096", "v"),
        "B3_fixed_threshold": ("fixed τ=0.8", "#dd6b20", "s"),
        "B2_hand_tuned": ("hand-tuned (FrugalGPT-style)", "#38a169", "D"),
    }
    for name, (label, color, marker) in markers.items():
        pts = [r for r in rows if r["method"] == name]
        ax.scatter(
            [_f(r, "cost_per_1k") for r in pts],
            [_f(r, "accuracy") for r in pts],
            color=color, marker=marker, s=55, label=label, zorder=2,
        )

    ax.set_xlabel("cost per 1k rows (USD)")
    ax.set_ylabel("accuracy")
    ax.set_title("Cost–accuracy Pareto (synthetic workload)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "pareto.png", dpi=150)
    plt.close(fig)


def figure_coverage() -> None:
    """P2 — coverage validity: empirical vs nominal 1−α should hug the diagonal."""
    rows = _read(OUT / "coverage_sweep.csv")
    nominal = [_f(r, "nominal_coverage") for r in rows]
    empirical = [_f(r, "empirical_coverage") for r in rows]

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo = min(nominal + empirical) - 0.03
    ax.plot([lo, 1.0], [lo, 1.0], "--", color="#a0aec0", label="ideal (y = x)")
    ax.fill_between([lo, 1.0], [lo - 0.03, 0.97], [lo + 0.03, 1.03],
                    color="#c6f6d5", alpha=0.4, label="±3% target (SC2)")
    ax.plot(nominal, empirical, "-o", color="#2b6cb0", markersize=8,
            linewidth=2, label="conformal cascade", zorder=3)
    for r in rows:
        ax.annotate(f"α={_f(r, 'alpha'):.2f}",
                    (_f(r, "nominal_coverage"), _f(r, "empirical_coverage")),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)

    ax.set_xlabel("nominal coverage (1 − α)")
    ax.set_ylabel("empirical coverage (held-out)")
    ax.set_title("Coverage validity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "coverage_validity.png", dpi=150)
    plt.close(fig)


def figure_ablation() -> None:
    """P5 — optimizer ablation: savings vs naive with each pass disabled."""
    rows = _read(OUT / "optimizer_ablation.csv")
    labels = [r["config"] for r in rows]
    savings = [_f(r, "savings_vs_naive_pct") for r in rows]
    colors = ["#a0aec0" if lbl.startswith("none") else
              "#2b6cb0" if lbl == "full" else "#dd6b20" for lbl in labels]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar(labels, savings, color=colors)
    for b, s in zip(bars, savings, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, s + 1, f"{s:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("cost savings vs. naive (%)")
    ax.set_title("Optimizer ablation (disabling one pass)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "optimizer_ablation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    missing = [
        name for name, f in [
            ("pareto.csv", OUT / "pareto.csv"),
            ("coverage_sweep.csv", OUT / "coverage_sweep.csv"),
            ("optimizer_ablation.csv", OUT / "optimizer_ablation.csv"),
        ] if not f.exists()
    ]
    if missing:
        print(f"missing CSVs: {missing}\nrun `make full` first.", file=sys.stderr)
        raise SystemExit(1)

    figure_pareto()
    figure_coverage()
    figure_ablation()

    # Keep the committed README copies (docs/img/) in sync with the paper figures.
    docs = ROOT / "docs" / "img"
    docs.mkdir(parents=True, exist_ok=True)
    for name in ("pareto.png", "coverage_validity.png", "optimizer_ablation.png"):
        (docs / name).write_bytes((FIG / name).read_bytes())
        print(f"wrote {FIG / name}")
    print(f"synced README figures → {docs}")


if __name__ == "__main__":
    main()
