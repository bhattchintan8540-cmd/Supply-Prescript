"""Week 8 query overrides stay on the week 7 grid unless the point is off it."""

from week6.phase2_midpoint import sweep_phase2
from week7.recommend import recommend_midpoint
from week8.recommend import recommend_at


def test_omitted_params_match_the_demo_recommendation():
    rows = sweep_phase2()
    assert recommend_at(rows=rows)["winner_label"] == recommend_midpoint(rows)["winner_label"]


def test_on_grid_override_uses_that_cell():
    rows = sweep_phase2()
    result = recommend_at(budget=120_000, max_delay=8, rows=rows)
    cell = next(
        row
        for row in rows
        if row["budget_cap_usd"] == 120_000 and row["max_acceptable_delay_days"] == 8
    )
    assert result["budget_cap_usd"] == 120_000
    assert result["max_acceptable_delay_days"] == 8
    assert result["winner_label"] == cell["winner_label"]
    assert result["winner_cost_usd"] == cell["winner_cost_usd"]
    assert result["grid_cells"] == 9
