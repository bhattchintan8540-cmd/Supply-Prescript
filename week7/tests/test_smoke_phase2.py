"""Smoke check for week7/smoke_phase2.py."""

from week7.smoke_phase2 import run_smoke


def test_phase2_smoke_exports_and_names_winner():
    summary = run_smoke()
    assert summary["grid_cells"] == 9
    assert summary["winner_label"] in {
        "Air Freight",
        "Secondary Supplier",
        "Delay Launch",
    }
    assert summary["grid_path"].endswith("phase2_grid.csv")
    assert summary["json_path"].endswith("phase2_recommendation.json")
