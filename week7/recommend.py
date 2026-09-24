"""
Week 7 — stop Phase 2 at the demo operating point.

Uses the week 6 midpoint grid. Names the cheapest pure option that fits
the demo budget and delay ceiling, and counts how many neighboring cells
pick that same option. Does not write a decision or call retrain.

    python week7/recommend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week6.phase2_midpoint import (
    DEMO_BUDGET_USD,
    DEMO_MAX_DELAY_DAYS,
    SAMPLE,
    sweep_phase2,
)


def recommend_midpoint(rows: list[dict] | None = None) -> dict:
    grid = sweep_phase2() if rows is None else rows
    center = next(
        row
        for row in grid
        if row["budget_cap_usd"] == DEMO_BUDGET_USD
        and row["max_acceptable_delay_days"] == DEMO_MAX_DELAY_DAYS
    )
    winner = center["winner_label"]
    same = sum(1 for row in grid if winner is not None and row["winner_label"] == winner)
    return {
        "sku": SAMPLE["sku"],
        "budget_cap_usd": DEMO_BUDGET_USD,
        "max_acceptable_delay_days": DEMO_MAX_DELAY_DAYS,
        "winner_label": winner,
        "winner_cost_usd": center["winner_cost_usd"],
        "milp_feasible": center["milp_feasible"],
        "same_winner_cells": same,
        "grid_cells": len(grid),
    }


def format_recommendation(result: dict) -> str:
    winner = result["winner_label"] or "(none feasible)"
    cost = "n/a" if result["winner_cost_usd"] is None else f"${result['winner_cost_usd']:,.2f}"
    milp = "feasible" if result["milp_feasible"] else "infeasible"
    return "\n".join(
        [
            "Week 7 — Phase 2 midpoint recommendation",
            f"Shipment {result['sku']}",
            f"Demo point: budget ${result['budget_cap_usd']:,.0f}, "
            f"max delay {result['max_acceptable_delay_days']:.0f}d",
            f"Cheapest feasible pure option: {winner} ({cost})",
            f"MILP split at that point: {milp}",
            f"Same pure winner on {result['same_winner_cells']} of {result['grid_cells']} grid cells",
            "Stopped here: no decision write-back and no retrain.",
        ]
    )


def main() -> int:
    print(format_recommendation(recommend_midpoint()))
    print()
    print("WEEK7 PHASE2 MIDPOINT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
