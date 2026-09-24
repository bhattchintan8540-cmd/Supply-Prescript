"""Week 7 checks the midpoint recommendation against the week 6 grid."""

from week6.phase2_midpoint import DEMO_BUDGET_USD, DEMO_MAX_DELAY_DAYS, sweep_phase2
from week7.recommend import recommend_midpoint


def test_recommendation_matches_demo_cell():
    rows = sweep_phase2()
    result = recommend_midpoint(rows)
    center = next(
        row
        for row in rows
        if row["budget_cap_usd"] == DEMO_BUDGET_USD
        and row["max_acceptable_delay_days"] == DEMO_MAX_DELAY_DAYS
    )
    assert result["winner_label"] == center["winner_label"]
    assert result["winner_cost_usd"] == center["winner_cost_usd"]
    assert result["milp_feasible"] == center["milp_feasible"]
    assert result["grid_cells"] == len(rows)
    assert 1 <= result["same_winner_cells"] <= result["grid_cells"]
