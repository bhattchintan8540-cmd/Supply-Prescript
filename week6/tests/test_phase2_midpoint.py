"""Week 6 checks the Phase 2 midpoint grid without training a model."""

import csv
from pathlib import Path

from week6.phase2_midpoint import (
    BUDGETS_USD,
    DEMO_BUDGET_USD,
    DEMO_MAX_DELAY_DAYS,
    MAX_DELAYS_DAYS,
    evaluate_point,
    export_grid_csv,
    main,
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


def test_export_grid_csv_writes_nine_rows(tmp_path: Path):
    out = tmp_path / "phase2_grid.csv"
    rows = sweep_phase2()
    export_grid_csv(rows, path=out)
    with out.open(encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == len(rows)
    assert exported[0]["budget_cap_usd"] == str(rows[0]["budget_cap_usd"])


def test_csv_out_flag_writes_the_given_path(tmp_path: Path):
    dest = tmp_path / "custom_grid.csv"
    assert main(["--csv-out", str(dest)]) == 0
    with dest.open(encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 9


def test_tight_budget_can_leave_no_feasible_pure_option():
    """Very tight budget + delay ceiling → empty feasible set, not a crash."""
    row = evaluate_point(budget_cap_usd=1.0, max_acceptable_delay_days=0.0)
    assert row["winner_label"] is None
    assert row["winner_cost_usd"] is None
    assert all(not (opt["within_budget"] and opt["within_sla"]) for opt in row["options"])
    assert "milp_budget_relaxed" in row
    assert "milp_within_budget" in row


def test_eighty_thousand_cells_do_not_hide_a_relaxed_budget():
    """$80k is below every pure option. A feasible MILP must say if the cap was dropped."""
    rows = [row for row in sweep_phase2() if row["budget_cap_usd"] == 80_000]
    assert rows
    assert all(row["winner_label"] is None for row in rows)
    for row in rows:
        if row["milp_feasible"]:
            assert row["milp_budget_relaxed"] is True
            assert row["milp_within_budget"] is False
