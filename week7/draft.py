"""
In-memory Phase 2 decision draft.

Turns the midpoint recommendation into the shape of a decision record
without inserting a row, opening a session, or retraining.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week6.phase2_midpoint import (
    PREDICTED_DELAY_DAYS,
    PREDICTED_DELAY_PROBABILITY,
    SAMPLE,
    evaluate_point,
)
from week7.recommend import recommend_midpoint

DELAY_LAUNCH_LABEL = "Delay Launch"


def draft_decision(result: dict | None = None) -> dict:
    """Decision-shaped dict for the recommended operating point.

    ``persisted`` is always false. This helper does not write to the database.
    """
    recommendation = recommend_midpoint() if result is None else result
    budget = recommendation["budget_cap_usd"]
    delay = recommendation["max_acceptable_delay_days"]
    cell = next(
        (
            row
            for row in recommendation.get("grid") or []
            if row["budget_cap_usd"] == budget
            and row["max_acceptable_delay_days"] == delay
            and row.get("options")
        ),
        None,
    )
    if cell is None:
        cell = evaluate_point(budget, delay)
    delay_launch = next(
        (opt for opt in cell["options"] if opt["label"] == DELAY_LAUNCH_LABEL),
        None,
    )
    return {
        "shipment_sku": recommendation["sku"],
        "predicted_delay_days": PREDICTED_DELAY_DAYS,
        "predicted_delay_probability": PREDICTED_DELAY_PROBABILITY,
        "shipment_features": dict(SAMPLE),
        "options": cell["options"],
        "chosen_option_label": recommendation["winner_label"],
        "predicted_cost_usd": recommendation["winner_cost_usd"],
        "no_action_cost_usd": None if delay_launch is None else delay_launch["cost_usd"],
        "budget_cap_usd": budget,
        "max_acceptable_delay_days": delay,
        "actual_cost_usd": None,
        "actual_delay_days": None,
        "persisted": False,
    }
