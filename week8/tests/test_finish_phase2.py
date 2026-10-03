"""The Phase 2 completion check rejects a broken smoke summary."""

from week8.finish_phase2 import verify_summary
from week8.smoke_phase2 import run_smoke


def test_live_smoke_passes_the_completion_checks():
    assert verify_summary(run_smoke()) == []


def test_persisted_draft_fails_the_completion_check():
    summary = {
        "grid_cells": 9,
        "winner_label": "Secondary Supplier",
        "draft_persisted": True,
        "draft_sku": "MICROCHIP-A2",
        "grid_path": "data/phase2_grid.csv",
        "json_path": "data/phase2_recommendation.json",
    }
    errors = verify_summary(summary)
    assert any("persisted" in item for item in errors)
