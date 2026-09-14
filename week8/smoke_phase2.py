"""
Week 8 smoke — export the grid, name the recommendation, draft in memory.

Does not train a model, call FastAPI, or insert a decision row.

    python week8/smoke_phase2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week6.phase2_midpoint import export_grid_csv, sweep_phase2
from week7.draft import draft_decision
from week7.recommend import persist_recommendation, recommend_midpoint


def run_smoke() -> dict:
    rows = sweep_phase2()
    grid_path = export_grid_csv(rows)
    result = recommend_midpoint(rows)
    json_path = persist_recommendation(result)
    draft = draft_decision(result)
    return {
        "grid_cells": len(rows),
        "winner_label": result["winner_label"],
        "grid_path": str(grid_path),
        "json_path": str(json_path),
        "draft_sku": draft["shipment_sku"],
        "draft_persisted": draft["persisted"],
    }


def main() -> int:
    summary = run_smoke()
    print(
        f"grid_cells={summary['grid_cells']} "
        f"winner={summary['winner_label']} "
        f"csv={summary['grid_path']} "
        f"json={summary['json_path']} "
        f"draft_persisted={summary['draft_persisted']}"
    )
    print("WEEK8 PHASE2 FINISH OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
