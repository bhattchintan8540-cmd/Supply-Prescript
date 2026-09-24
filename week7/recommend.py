"""
Week 7 — stop Phase 2 at the demo operating point.

Uses the week 6 midpoint grid. Names the cheapest pure option that fits
the demo budget and delay ceiling, and counts how many neighboring cells
pick that same option. Does not write a decision or call retrain.

    python week7/recommend.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week6.phase2_midpoint import (
    DEMO_BUDGET_USD,
    DEMO_MAX_DELAY_DAYS,
    SAMPLE,
    sweep_phase2,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
RECOMMENDATION_JSON_PATH = ROOT_DIR / "data" / "phase2_recommendation.json"


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
        "grid": grid,
    }


def format_grid_table(grid: list[dict]) -> str:
    """Pretty-print every midpoint cell as an aligned table."""
    lines = [
        f"{'budget':>12}  {'max delay':>9}  {'pure winner':<22}  {'cost':>12}  milp",
        "-" * 72,
    ]
    for row in grid:
        cost = "—" if row["winner_cost_usd"] is None else f"${row['winner_cost_usd']:,.2f}"
        winner = row["winner_label"] or "(none feasible)"
        milp = "feasible" if row["milp_feasible"] else "infeasible"
        marker = " *" if (
            row["budget_cap_usd"] == DEMO_BUDGET_USD
            and row["max_acceptable_delay_days"] == DEMO_MAX_DELAY_DAYS
        ) else ""
        lines.append(
            f"${row['budget_cap_usd']:>10,.0f}  "
            f"{row['max_acceptable_delay_days']:>8.0f}d  "
            f"{winner:<22}  {cost:>12}  {milp}{marker}"
        )
    lines.append("(* = demo operating point)")
    return "\n".join(lines)


def format_recommendation(result: dict) -> str:
    winner = result["winner_label"] or "(none feasible)"
    cost = "n/a" if result["winner_cost_usd"] is None else f"${result['winner_cost_usd']:,.2f}"
    milp = "feasible" if result["milp_feasible"] else "infeasible"
    header = "\n".join(
        [
            "Week 7 — Phase 2 midpoint recommendation",
            f"Shipment {result['sku']}",
            f"Demo point: budget ${result['budget_cap_usd']:,.0f}, "
            f"max delay {result['max_acceptable_delay_days']:.0f}d",
            f"Cheapest feasible pure option: {winner} ({cost})",
            f"MILP split at that point: {milp}",
            f"Same pure winner on {result['same_winner_cells']} of {result['grid_cells']} grid cells",
            "",
            format_grid_table(result.get("grid") or []),
            "",
            "Stopped here: no decision write-back and no retrain.",
        ]
    )
    return header


def _json_ready(result: dict) -> dict:
    """Drop nested option dicts so the JSON stays a compact summary."""
    payload = {key: value for key, value in result.items() if key != "grid"}
    payload["grid"] = [
        {
            "budget_cap_usd": row["budget_cap_usd"],
            "max_acceptable_delay_days": row["max_acceptable_delay_days"],
            "winner_label": row["winner_label"],
            "winner_cost_usd": row["winner_cost_usd"],
            "milp_feasible": row["milp_feasible"],
        }
        for row in result.get("grid") or []
    ]
    return payload


def persist_recommendation(
    result: dict,
    path: Path = RECOMMENDATION_JSON_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
    return path


def main() -> int:
    result = recommend_midpoint()
    json_path = persist_recommendation(result)
    print(format_recommendation(result))
    print()
    print(f"Wrote {json_path}")
    print("WEEK7 PHASE2 MIDPOINT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
