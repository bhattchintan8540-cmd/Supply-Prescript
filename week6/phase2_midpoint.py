"""
Week 6 — Phase 2, stopped at the midpoint.

Week 2 already prices air freight, the secondary supplier, delay launch,
and the PuLP split. This week does not add channels or a new optimizer.
It only walks a short budget × max-delay grid around the closed-loop
demo shipment and records which pure option stays inside both limits.

    python week6/phase2_midpoint.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week2.solver import pure_options, solve_optimal_allocation

ROOT_DIR = Path(__file__).resolve().parent.parent
GRID_CSV_PATH = ROOT_DIR / "data" / "phase2_grid.csv"

# Same shipment and prediction the week 5 smoke loop uses, so this run
# does not need a fresh training pass.
SAMPLE = {
    "sku": "MICROCHIP-A2",
    "supplier": "Delta Cove Electronics",
    "origin_region": "Asia Pacific",
    "distance_km": 9500.0,
    "historical_avg_lead_time_days": 18.0,
    "order_quantity": 6000,
    "unit_cost_usd": 14.2,
    "is_peak_season": True,
}
PREDICTED_DELAY_DAYS = 6.0
PREDICTED_DELAY_PROBABILITY = 0.8

# Midpoint grid: one step under the demo point, the demo point, one step over.
BUDGETS_USD = (80_000.0, 100_000.0, 120_000.0)
MAX_DELAYS_DAYS = (3.0, 5.0, 8.0)
DEMO_BUDGET_USD = 100_000.0
DEMO_MAX_DELAY_DAYS = 5.0


def evaluate_point(budget_cap_usd: float, max_acceptable_delay_days: float) -> dict:
    """One Phase 2 solve at a single budget and delay ceiling."""
    options = pure_options(
        unit_cost_usd=SAMPLE["unit_cost_usd"],
        order_quantity=SAMPLE["order_quantity"],
        predicted_delay_days=PREDICTED_DELAY_DAYS,
        budget_cap_usd=budget_cap_usd,
        predicted_delay_probability=PREDICTED_DELAY_PROBABILITY,
        max_acceptable_delay_days=max_acceptable_delay_days,
    )
    blend = solve_optimal_allocation(
        unit_cost_usd=SAMPLE["unit_cost_usd"],
        order_quantity=SAMPLE["order_quantity"],
        predicted_delay_days=PREDICTED_DELAY_DAYS,
        budget_cap_usd=budget_cap_usd,
        max_acceptable_delay_days=max_acceptable_delay_days,
        predicted_delay_probability=PREDICTED_DELAY_PROBABILITY,
    )
    feasible = [opt for opt in options if opt["within_budget"] and opt["within_sla"]]
    winner = min(feasible, key=lambda opt: opt["cost_usd"]) if feasible else None
    return {
        "budget_cap_usd": budget_cap_usd,
        "max_acceptable_delay_days": max_acceptable_delay_days,
        "options": options,
        "winner_label": None if winner is None else winner["label"],
        "winner_cost_usd": None if winner is None else winner["cost_usd"],
        "milp_status": blend.get("status"),
        "milp_feasible": not bool(blend.get("infeasible")),
    }


def sweep_phase2(
    budgets: tuple[float, ...] | None = None,
    max_delays: tuple[float, ...] | None = None,
) -> list[dict]:
    """Budget × delay cells. Defaults to the 3×3 midpoint grid."""
    budget_values = BUDGETS_USD if budgets is None else budgets
    delay_values = MAX_DELAYS_DAYS if max_delays is None else max_delays
    return [
        evaluate_point(budget, max_delay)
        for budget in budget_values
        for max_delay in delay_values
    ]


def _parse_float_list(raw: str) -> tuple[float, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one number")
    return tuple(float(part) for part in parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Week 6 Phase 2 midpoint budget × max-delay sweep",
    )
    parser.add_argument(
        "--budgets",
        type=_parse_float_list,
        default=None,
        help="Comma-separated budget caps in USD (default: 80000,100000,120000)",
    )
    parser.add_argument(
        "--max-delays",
        type=_parse_float_list,
        default=None,
        help="Comma-separated max delay days (default: 3,5,8)",
    )
    return parser


def format_report(rows: list[dict]) -> str:
    lines = [
        "Week 6 — Phase 2 midpoint (budget x max delay)",
        f"Shipment {SAMPLE['sku']}  delay {PREDICTED_DELAY_DAYS}d  "
        f"P(delay)={PREDICTED_DELAY_PROBABILITY:.0%}",
        "-" * 72,
        f"{'budget':>12}  {'max delay':>9}  {'pure winner':<22}  {'cost':>12}  milp",
    ]
    for row in rows:
        cost = "—" if row["winner_cost_usd"] is None else f"${row['winner_cost_usd']:,.2f}"
        winner = row["winner_label"] or "(none feasible)"
        milp = "feasible" if row["milp_feasible"] else "infeasible"
        lines.append(
            f"${row['budget_cap_usd']:>10,.0f}  "
            f"{row['max_acceptable_delay_days']:>8.0f}d  "
            f"{winner:<22}  {cost:>12}  {milp}"
        )
    return "\n".join(lines)


def export_grid_csv(rows: list[dict], path: Path = GRID_CSV_PATH) -> Path:
    """Write the nine midpoint cells to CSV under data/ (gitignored)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "budget_cap_usd",
        "max_acceptable_delay_days",
        "winner_label",
        "winner_cost_usd",
        "milp_status",
        "milp_feasible",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = sweep_phase2(budgets=args.budgets, max_delays=args.max_delays)
    csv_path = export_grid_csv(rows)
    print(format_report(rows))
    print()
    print(f"Wrote {csv_path}")
    print("WEEK6 PHASE2 MIDPOINT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
