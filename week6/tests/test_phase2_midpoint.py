"""Week 6 checks the Phase 2 midpoint grid without training a model."""

from week6.phase2_midpoint import (
    BUDGETS_USD,
    DEMO_BUDGET_USD,
    DEMO_MAX_DELAY_DAYS,
    MAX_DELAYS_DAYS,
    sweep_phase2,
)


def test_midpoint_grid_is_three_by_three():
    rows = sweep_phase2()
    assert len(rows) == len(BUDGETS_USD) * len(MAX_DELAYS_DAYS)
    assert {row["budget_cap_usd"] for row in rows} == set(BUDGETS_USD)
    assert {row["max_acceptable_delay_days"] for row in rows} == set(MAX_DELAYS_DAYS)


def test_demo_point_names_a_feasible_pure_option():
    rows = sweep_phase2()
    center = next(
        row
        for row in rows
        if row["budget_cap_usd"] == DEMO_BUDGET_USD
        and row["max_acceptable_delay_days"] == DEMO_MAX_DELAY_DAYS
    )
    assert center["winner_label"] in {"Air Freight", "Secondary Supplier", "Delay Launch"}
    chosen = next(opt for opt in center["options"] if opt["label"] == center["winner_label"])
    assert chosen["within_budget"] is True
    assert chosen["within_sla"] is True
    assert center["winner_cost_usd"] == chosen["cost_usd"]
