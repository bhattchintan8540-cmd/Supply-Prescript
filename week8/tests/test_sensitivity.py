"""Neighboring-cell sensitivity uses the week 7 recommendation grid."""

from week7.recommend import recommend_midpoint
from week8.sensitivity import sensitivity_summary


def test_sensitivity_counts_agreement_and_cost_spread():
    result = recommend_midpoint()
    summary = sensitivity_summary(result)
    costs = [
        row["winner_cost_usd"]
        for row in result["grid"]
        if row["winner_cost_usd"] is not None
    ]
    assert summary["winner_label"] == result["winner_label"]
    assert summary["grid_cells"] == result["grid_cells"]
    assert summary["same_winner_cells"] == result["same_winner_cells"]
    assert 1 <= summary["same_winner_cells"] <= summary["grid_cells"]
    assert summary["cost_min_usd"] == min(costs)
    assert summary["cost_max_usd"] == max(costs)
    assert summary["cost_spread_usd"] == round(max(costs) - min(costs), 2)
