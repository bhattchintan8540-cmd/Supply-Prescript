"""
Phase 2 completion check. Does not need GNU make.

Runs the week 6 grid, the week 7 recommendation, the in-memory draft,
and the Phase 2 API routes. Exits 0 only when those checks pass.

    python week8/finish_phase2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week2.solver import (
    AIR_FREIGHT_HANDLING_FEE,
    AIR_FREIGHT_RESULTING_DELAY_DAYS,
    AIR_FREIGHT_SURCHARGE_PER_UNIT,
    HOLDING_COST_PER_UNIT_PER_DAY,
    SECONDARY_SUPPLIER_DELAY_FACTOR,
    SECONDARY_SUPPLIER_HANDLING_FEE,
    SECONDARY_SUPPLIER_PREMIUM_PCT,
)
from week6.phase2_midpoint import (
    DEMO_BUDGET_USD,
    DEMO_MAX_DELAY_DAYS,
    PREDICTED_DELAY_DAYS,
    PREDICTED_DELAY_PROBABILITY,
    SAMPLE,
)
from week7.recommend import recommend_midpoint
from week8.sensitivity import sensitivity_summary
from week8.smoke_phase2 import run_smoke

PURE_OPTIONS = {"Air Freight", "Secondary Supplier", "Delay Launch"}


def verify_summary(summary: dict) -> list[str]:
    """Return human-readable failures. An empty list means the smoke passed."""
    errors: list[str] = []
    if summary.get("grid_cells") != 9:
        errors.append(f"expected 9 grid cells, got {summary.get('grid_cells')}")
    winner = summary.get("winner_label")
    if winner not in PURE_OPTIONS:
        errors.append(f"demo winner {winner!r} is not a pure option")
    if summary.get("draft_persisted") is not False:
        errors.append("decision draft was persisted; Phase 2 must stay in memory")
    if not str(summary.get("draft_sku") or ""):
        errors.append("draft is missing a shipment sku")
    if not str(summary.get("grid_path") or "").endswith("phase2_grid.csv"):
        errors.append(f"grid CSV path is unexpected: {summary.get('grid_path')}")
    if not str(summary.get("json_path") or "").endswith("phase2_recommendation.json"):
        errors.append(f"recommendation JSON path is unexpected: {summary.get('json_path')}")
    return errors


def verify_api() -> list[str]:
    """Hit the Phase 2 routes in-process. Does not insert a decision row."""
    from fastapi.testclient import TestClient

    from week1.database import init_db
    from week3.main import app

    init_db()
    errors: list[str] = []
    try:
        client_cm = TestClient(app)
        client = client_cm.__enter__()
    except Exception as exc:
        return [f"API client failed to start: {exc}"]
    try:
        before = client.get("/decisions")
        if before.status_code != 200:
            return [f"GET /decisions returned {before.status_code}"]
        before_count = len(before.json())

        grid = client.get("/phase2/grid")
        if grid.status_code != 200 or grid.json().get("grid_cells") != 9:
            errors.append("GET /phase2/grid did not return nine cells")

        recommend = client.get("/phase2/recommend")
        if recommend.status_code != 200:
            errors.append(f"GET /phase2/recommend returned {recommend.status_code}")
        else:
            body = recommend.json()
            if body.get("budget_cap_usd") != DEMO_BUDGET_USD:
                errors.append("recommend budget is not the $100,000 demo point")
            if body.get("max_acceptable_delay_days") != DEMO_MAX_DELAY_DAYS:
                errors.append("recommend max delay is not the 5-day demo point")
            if body.get("winner_label") not in PURE_OPTIONS:
                errors.append("recommend winner is not a pure option")

        override = client.get(
            "/phase2/recommend",
            params={"budget": 120_000, "max_delay": 8},
        )
        if override.status_code != 200:
            errors.append(f"recommend override returned {override.status_code}")
        elif override.json().get("budget_cap_usd") != 120_000:
            errors.append("budget query param was ignored")

        draft = client.post("/phase2/draft-decision", json={})
        if draft.status_code != 200 or draft.json().get("persisted") is not False:
            errors.append("POST /phase2/draft-decision did not stay a preview")

        after = client.get("/decisions")
        if after.status_code != 200 or len(after.json()) != before_count:
            errors.append("draft preview changed the decisions table")
    except Exception as exc:
        errors.append(f"API check failed: {exc}")
    finally:
        client_cm.__exit__(None, None, None)
    return errors


def _print_parameters(summary: dict) -> None:
    result = recommend_midpoint()
    spread = sensitivity_summary(result)
    print("PARAMETERS")
    print(f"  sku={SAMPLE['sku']}  qty={SAMPLE['order_quantity']}  unit_cost=${SAMPLE['unit_cost_usd']}")
    print(
        f"  predicted_delay={PREDICTED_DELAY_DAYS}d  "
        f"P(delay)={PREDICTED_DELAY_PROBABILITY:.0%}"
    )
    print(
        f"  demo_budget=${DEMO_BUDGET_USD:,.0f}  "
        f"demo_max_delay={DEMO_MAX_DELAY_DAYS:.0f}d  "
        f"grid=80000/100000/120000 x 3/5/8 days"
    )
    print(
        f"  air_surcharge=${AIR_FREIGHT_SURCHARGE_PER_UNIT}/unit  "
        f"air_fee=${AIR_FREIGHT_HANDLING_FEE:,.0f}  "
        f"air_delay={AIR_FREIGHT_RESULTING_DELAY_DAYS:.0f}d"
    )
    print(
        f"  secondary_premium={SECONDARY_SUPPLIER_PREMIUM_PCT:.0%}  "
        f"secondary_fee=${SECONDARY_SUPPLIER_HANDLING_FEE:,.0f}  "
        f"delay_factor={SECONDARY_SUPPLIER_DELAY_FACTOR}"
    )
    print(f"  holding=${HOLDING_COST_PER_UNIT_PER_DAY}/unit/day")
    print(
        f"  winner={summary['winner_label']}  "
        f"same_winner_cells={spread['same_winner_cells']}/{spread['grid_cells']}  "
        f"cost_spread=${spread['cost_spread_usd']}"
    )
    print("LIMITATIONS")
    print("  Stand-in freight and holding rates are not contracted prices.")
    print("  Secondary supplier is a scenario, not a qualified supplier pick.")
    print("  The $80,000 row has no feasible pure option on this shipment.")
    print("  The draft is in memory only. Phase 2 does not write a decision or retrain.")
    print("DEBUG")
    print("  If WEEK6 fails, rerun: python week6/phase2_midpoint.py")
    print("  If the API check fails, confirm uvicorn is not required; this script uses TestClient.")
    print("  Tight budget + delay is an empty feasible set, not a crash.")


def main() -> int:
    summary = run_smoke()
    errors = verify_summary(summary)
    errors.extend(verify_api())
    print(
        f"grid_cells={summary['grid_cells']} "
        f"winner={summary['winner_label']} "
        f"draft_persisted={summary['draft_persisted']}"
    )
    _print_parameters(summary)
    if errors:
        print("PHASE 2 CHECKS FAILED")
        for item in errors:
            print(f"  - {item}")
        return 1
    print("WEEK8 PHASE2 COMPLETE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
