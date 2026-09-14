"""Smoke check for week8/smoke_phase2.py."""

from week8.smoke_phase2 import run_smoke


def test_phase2_finish_exports_recommendation_and_draft():
    summary = run_smoke()
    assert summary["grid_cells"] == 9
    assert summary["winner_label"] in {
        "Air Freight",
        "Secondary Supplier",
        "Delay Launch",
    }
    assert summary["grid_path"].endswith("phase2_grid.csv")
    assert summary["json_path"].endswith("phase2_recommendation.json")
    assert summary["draft_sku"]
    assert summary["draft_persisted"] is False
