"""Week 7 checks the midpoint recommendation against the week 6 grid."""

import json
from pathlib import Path

from week6.phase2_midpoint import DEMO_BUDGET_USD, DEMO_MAX_DELAY_DAYS, sweep_phase2
from week7.recommend import (
    format_grid_table,
    format_recommendation,
    persist_recommendation,
    recommend_midpoint,
)


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


def test_recommendation_print_includes_grid_table():
    result = recommend_midpoint()
    table = format_grid_table(result["grid"])
    text = format_recommendation(result)
    assert "pure winner" in table
    assert "* = demo operating point" in table
    assert table in text


def test_persist_recommendation_writes_json(tmp_path: Path):
    out = tmp_path / "phase2_recommendation.json"
    result = recommend_midpoint()
    persist_recommendation(result, path=out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["winner_label"] == result["winner_label"]
    assert len(payload["grid"]) == result["grid_cells"]
