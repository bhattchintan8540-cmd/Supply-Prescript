"""Compare neighboring Phase 2 cells: agreement and cost spread."""
from __future__ import annotations


def sensitivity_summary(result: dict) -> dict:
    """How stable the pure winner is across the midpoint grid."""
    grid = result.get("grid") or []
    winner = result.get("winner_label")
    costs = [
        row["winner_cost_usd"]
        for row in grid
        if row.get("winner_cost_usd") is not None
    ]
    same = sum(1 for row in grid if winner is not None and row.get("winner_label") == winner)
    spread = None if not costs else round(max(costs) - min(costs), 2)
    return {
        "winner_label": winner,
        "same_winner_cells": same,
        "grid_cells": len(grid),
        "cost_min_usd": None if not costs else min(costs),
        "cost_max_usd": None if not costs else max(costs),
        "cost_spread_usd": spread,
    }
