"""Per-row execution trace (FR-10.2).

Writes one JSON object per row to a ``.jsonl`` file: the final answer, the tier that
produced it, the full escalation history (every tier visited, its answer, confidence, and
cost), and the row's total cost. This is the audit trail that lets any headline number be
traced back to individual rows (NFR-1).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from semopt.cascade.cascade import CascadeResult


def result_to_dict(row_index: int, r: CascadeResult) -> dict[str, object]:
    """Serialize one :class:`CascadeResult` (FR-6.4) into a JSON-ready dict."""
    return {
        "row": row_index,
        "final_answer": r.value,
        "final_tier": r.final_tier,
        "confidence": round(r.confidence, 6),
        "total_cost_usd": round(r.total_cost_usd, 8),
        "tokens_in": r.tokens_in,
        "tokens_out": r.tokens_out,
        "escalation_history": [
            {
                "tier_index": v.tier_index,
                "model_id": v.model_id,
                "answer": v.value,
                "confidence": round(v.score, 6),
                "method": v.method,
                "accepted": v.accepted,
                "cost_usd": round(v.cost_usd, 8),
            }
            for v in r.history
        ],
    }


def write_trace(results: Sequence[CascadeResult], path: str | Path) -> int:
    """Write per-row traces as JSONL to ``path``. Returns the number of rows written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for i, r in enumerate(results):
            fh.write(json.dumps(result_to_dict(i, r)) + "\n")
    return len(results)
