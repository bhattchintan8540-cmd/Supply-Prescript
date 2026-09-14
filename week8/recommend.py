"""
Week 8 — Phase 2 finish: recommend at the demo cell or a query override.

Week 7 names the demo operating point and stops. This module accepts an
optional budget and delay ceiling, including a point that is not on the
3×3 grid. It does not write a decision or retrain.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week6.phase2_midpoint import evaluate_point
from week7.recommend import recommend_midpoint


def recommend_at(
    budget: float | None = None,
    max_delay: float | None = None,
    rows: list[dict] | None = None,
) -> dict:
    """Midpoint recommendation, optionally at another budget or delay ceiling.

    Omitted values stay on the demo cell. An off-grid pair is solved once
    and the nine-cell grid is kept for context.
    """
    result = recommend_midpoint(rows)
    if budget is None and max_delay is None:
        return result
    budget_cap = result["budget_cap_usd"] if budget is None else budget
    delay_cap = result["max_acceptable_delay_days"] if max_delay is None else max_delay
    match = next(
        (
            row
            for row in result["grid"]
            if row["budget_cap_usd"] == budget_cap
            and row["max_acceptable_delay_days"] == delay_cap
        ),
        None,
    )
    if match is None:
        match = evaluate_point(budget_cap, delay_cap)
    winner = match["winner_label"]
    return {
        **result,
        "budget_cap_usd": budget_cap,
        "max_acceptable_delay_days": delay_cap,
        "winner_label": winner,
        "winner_cost_usd": match["winner_cost_usd"],
        "milp_feasible": match["milp_feasible"],
        "same_winner_cells": sum(
            1
            for row in result["grid"]
            if winner is not None and row["winner_label"] == winner
        ),
    }
