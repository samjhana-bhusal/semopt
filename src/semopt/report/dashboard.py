"""Cascade dashboard — a self-contained HTML report per experiment (FR-10.3).

Summarizes one cascade run: the tier distribution (what fraction of rows each tier
served), the cost breakdown (total, and per-tier), and — when ground truth is supplied —
coverage vs. the α target. No external assets; the HTML inlines its own CSS so it opens
anywhere.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from semopt.cascade.cascade import CascadeResult


@dataclass
class DashboardStats:
    n_rows: int
    tier_fractions: dict[int, float]
    tier_model_ids: dict[int, str]
    total_cost_usd: float
    cost_by_tier: dict[int, float]
    empirical_coverage: float | None
    alpha: float | None


def compute_stats(
    results: Sequence[CascadeResult],
    *,
    correct: Sequence[bool] | None = None,
    alpha: float | None = None,
) -> DashboardStats:
    n = len(results)
    n_tiers = 1 + max((r.final_tier for r in results), default=0)
    tier_counts = {t: 0 for t in range(n_tiers)}
    cost_by_tier = {t: 0.0 for t in range(n_tiers)}
    tier_model_ids: dict[int, str] = {}
    total_cost = 0.0

    for r in results:
        tier_counts[r.final_tier] += 1
        total_cost += r.total_cost_usd
        for v in r.history:
            cost_by_tier.setdefault(v.tier_index, 0.0)
            cost_by_tier[v.tier_index] += v.cost_usd
            tier_model_ids.setdefault(v.tier_index, v.model_id)

    denom = max(n, 1)
    tier_fractions = {t: tier_counts[t] / denom for t in range(n_tiers)}
    coverage = (sum(1 for c in correct if c) / denom) if correct is not None else None

    return DashboardStats(
        n_rows=n,
        tier_fractions=tier_fractions,
        tier_model_ids=tier_model_ids,
        total_cost_usd=total_cost,
        cost_by_tier=cost_by_tier,
        empirical_coverage=coverage,
        alpha=alpha,
    )


_TIER_COLORS = ["#3C8C6C", "#B07CC9", "#C6793A", "#2E6CA6", "#8593A6"]


def _bar(fraction: float, color: str, label: str, value: str) -> str:
    pct = max(0.0, min(1.0, fraction)) * 100
    return (
        '<div class="row">'
        f'<div class="rl">{html.escape(label)}</div>'
        '<div class="track">'
        f'<div class="fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
        f'<div class="rv">{html.escape(value)}</div>'
        "</div>"
    )


def render_dashboard(stats: DashboardStats, *, title: str) -> str:
    tier_bars = "".join(
        _bar(
            stats.tier_fractions[t],
            _TIER_COLORS[t % len(_TIER_COLORS)],
            f"tier {t} · {stats.tier_model_ids.get(t, '?')}",
            f"{stats.tier_fractions[t] * 100:.1f}%",
        )
        for t in sorted(stats.tier_fractions)
    )
    max_tier_cost = max(stats.cost_by_tier.values(), default=0.0) or 1.0
    cost_bars = "".join(
        _bar(
            stats.cost_by_tier[t] / max_tier_cost,
            _TIER_COLORS[t % len(_TIER_COLORS)],
            f"tier {t}",
            f"${stats.cost_by_tier[t]:.6f}",
        )
        for t in sorted(stats.cost_by_tier)
    )

    coverage_block = ""
    if stats.empirical_coverage is not None:
        target = 1 - stats.alpha if stats.alpha is not None else None
        target_txt = f"{target:.2f}" if target is not None else "—"
        gap_txt = ""
        ok_class = ""
        if target is not None:
            gap = stats.empirical_coverage - target
            ok_class = "ok" if abs(gap) <= 0.03 else "warn"
            gap_txt = f"<span class='chip {ok_class}'>gap {gap:+.3f}</span>"
        coverage_block = (
            '<section><h2>Coverage vs. target</h2>'
            '<div class="cards">'
            f'<div class="card"><div class="n">{stats.empirical_coverage:.3f}</div>'
            '<div class="l">empirical accuracy</div></div>'
            f'<div class="card"><div class="n">{target_txt}</div>'
            f'<div class="l">target (1 − α){", α=" + format(stats.alpha, ".2f") if stats.alpha is not None else ""}</div></div>'
            f'<div class="card">{gap_txt or "—"}<div class="l">within ±3%?</div></div>'
            "</div></section>"
        )

    escalated = 1.0 - stats.tier_fractions.get(0, 0.0)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ --ink:#18202E; --soft:#4C5A6E; --line:#D3DBE7; --panel:#FBFCFE; --bg:#EEF2F7; --accent:#2E6CA6; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; line-height:1.5; }}
  .wrap {{ max-width:760px; margin:0 auto; padding:32px 22px 60px; }}
  h1 {{ font-size:1.5rem; margin:0 0 4px; letter-spacing:-0.02em; }}
  .sub {{ color:var(--soft); font-family:ui-monospace,Menlo,monospace; font-size:13px; margin-bottom:26px; }}
  h2 {{ font-size:1.05rem; margin:26px 0 12px; }}
  section {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px 20px; margin-bottom:16px; }}
  .row {{ display:grid; grid-template-columns:170px 1fr 82px; align-items:center; gap:12px; margin:8px 0; }}
  .rl {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; color:var(--soft); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .track {{ background:#E7ECF3; border:1px solid var(--line); border-radius:6px; height:20px; overflow:hidden; }}
  .fill {{ height:100%; border-radius:5px 0 0 5px; }}
  .rv {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; text-align:right; font-variant-numeric:tabular-nums; }}
  .cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
  .card {{ background:#F1F4F9; border:1px solid var(--line); border-radius:10px; padding:14px; }}
  .card .n {{ font-family:ui-monospace,Menlo,monospace; font-size:1.5rem; font-weight:700; }}
  .card .l {{ font-size:0.78rem; color:var(--soft); margin-top:2px; }}
  .chip {{ font-family:ui-monospace,Menlo,monospace; font-size:0.9rem; font-weight:700; padding:3px 10px; border-radius:20px; display:inline-block; }}
  .chip.ok {{ color:#2f7a56; background:#d6f0e2; }}
  .chip.warn {{ color:#9a5a20; background:#f6e2cd; }}
  .foot {{ color:var(--soft); font-size:0.8rem; margin-top:20px; }}
</style></head>
<body><div class="wrap">
  <h1>{html.escape(title)}</h1>
  <div class="sub">{stats.n_rows} rows · {escalated * 100:.1f}% escalated beyond tier 0 · total ${stats.total_cost_usd:.6f}</div>
  <section><h2>Tier distribution</h2>{tier_bars}</section>
  <section><h2>Cost breakdown by tier</h2>{cost_bars}</section>
  {coverage_block}
  <div class="foot">Generated by semopt (FR-10.3). Every number here is reproducible from the same run's <code>trace.jsonl</code> (FR-10.2).</div>
</div></body></html>"""


def write_dashboard(
    results: Sequence[CascadeResult],
    path: str | Path,
    *,
    title: str = "Cascade run",
    correct: Sequence[bool] | None = None,
    alpha: float | None = None,
) -> DashboardStats:
    """Write the HTML dashboard for a cascade run; returns the computed stats."""
    stats = compute_stats(results, correct=correct, alpha=alpha)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_dashboard(stats, title=title))
    return stats
