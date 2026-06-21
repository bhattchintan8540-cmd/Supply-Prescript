"""
Week 1 - mock data generator.

Real lead-time history is exactly the kind of thing you can't just
download, so this fabricates a plausible three-year shipment log:
a handful of suppliers/regions with different baseline reliability,
a peak-season effect, and a true calendar "bad quarter" for one
supplier so the delay model has a temporal pattern to learn — and so
train/validation/test can be split by shipment date rather than at
random.

Important honesty note for interviews: every relationship below is
programmed into the synthetic environment. Model metrics therefore
show that XGBoost recovers those relationships, not that the same
AUC/MAE would hold on a real carrier/supplier network.

Run directly to (re)build data/shipments.csv:
    python week1/generate_mock_data.py
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "shipments.csv"
RNG_SEED = 42

# Three calendar years of synthetic history. Temporal validation in
# DelayModel.fit() trains on earlier shipments and tests on later ones.
HISTORY_START = date(2023, 1, 1)
HISTORY_END = date(2025, 12, 31)

# True "bad quarter": Delta Cove deteriorates for a contiguous period,
# not a periodic record-index shock. That matches the business story
# ("supplier had a bad quarter") and gives temporal validation something
# meaningful to detect.
DELTA_COVE_BAD_QUARTER_START = date(2024, 7, 1)
DELTA_COVE_BAD_QUARTER_END = date(2024, 9, 30)

SUPPLIERS = {
    # name -> (baseline reliability, region)
    "NovaChip Manufacturing": (0.92, "Asia Pacific"),
    "Redline Components": (0.85, "Asia Pacific"),
    "Baltic Precision Parts": (0.88, "Europe"),
    "Meridian Fasteners": (0.95, "North America"),
    "Delta Cove Electronics": (0.78, "Asia Pacific"),
}
SKUS = ["MICROCHIP-A2", "RESISTOR-PACK", "CASING-ALU", "SENSOR-IR", "PCB-STD-4L"]

REGION_BASE_DISTANCE_KM = {
    "Asia Pacific": 9200,
    "Europe": 6800,
    "North America": 2400,
}


def _month_is_peak(month: int) -> bool:
    # Nov/Dec holiday ramp - matches the seasonality bump used elsewhere
    # in the portfolio (see the MetricMind seed data).
    return month in (11, 12)


def _random_shipment_date(rng: random.Random) -> date:
    span_days = (HISTORY_END - HISTORY_START).days
    return HISTORY_START + timedelta(days=rng.randint(0, span_days))


def _in_bad_quarter(supplier: str, shipment_date: date) -> bool:
    return (
        supplier == "Delta Cove Electronics"
        and DELTA_COVE_BAD_QUARTER_START <= shipment_date <= DELTA_COVE_BAD_QUARTER_END
    )


def build(n_rows: int = 4000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    rows = []
    supplier_names = list(SUPPLIERS.keys())

    for _ in range(n_rows):
        supplier = py_rng.choice(supplier_names)
        reliability, region = SUPPLIERS[supplier]
        sku = py_rng.choice(SKUS)
        shipment_date = _random_shipment_date(py_rng)
        month = shipment_date.month
        peak = _month_is_peak(month)

        distance = REGION_BASE_DISTANCE_KM[region] * rng.uniform(0.85, 1.15)
        avg_lead_time = 7 + distance / 900 + rng.normal(0, 1.2)
        avg_lead_time = max(avg_lead_time, 3)

        qty = int(rng.integers(500, 20_000))
        unit_cost = round(float(rng.uniform(0.8, 40)), 2)

        # Calendar-bounded supplier deterioration (a real quarter), not a
        # modulo-index shock on the row number.
        shock = 0.0
        if _in_bad_quarter(supplier, shipment_date):
            shock = rng.uniform(4, 11)

        peak_penalty = rng.uniform(1.5, 4.0) if peak else 0.0
        base_delay = max(0.0, rng.normal((1 - reliability) * 18, 2.5))
        actual_delay = round(max(0.0, base_delay + peak_penalty + shock), 1)

        rows.append(
            {
                "shipment_date": shipment_date.isoformat(),
                "sku": sku,
                "supplier": supplier,
                "origin_region": region,
                "distance_km": round(distance, 1),
                "historical_avg_lead_time_days": round(avg_lead_time, 1),
                "order_quantity": qty,
                "unit_cost_usd": unit_cost,
                "is_peak_season": peak,
                "actual_delay_days": actual_delay,
            }
        )

    return pd.DataFrame(rows).sort_values("shipment_date").reset_index(drop=True)


def main() -> None:
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(df)} rows -> {OUT_PATH}")
    print(f"date range: {df['shipment_date'].min()} .. {df['shipment_date'].max()}")
    bad = df[
        (df["supplier"] == "Delta Cove Electronics")
        & (df["shipment_date"] >= DELTA_COVE_BAD_QUARTER_START.isoformat())
        & (df["shipment_date"] <= DELTA_COVE_BAD_QUARTER_END.isoformat())
    ]
    print(f"Delta Cove bad-quarter rows: {len(bad)} "
          f"({DELTA_COVE_BAD_QUARTER_START} .. {DELTA_COVE_BAD_QUARTER_END})")
    print(df["actual_delay_days"].describe())
    seeded = seed_shipments_table(df)
    print(f"seeded shipments table: {seeded} rows")


def seed_shipments_table(df: pd.DataFrame) -> int:
    """Write the generated history into the Shipment ORM table.

    Training still reads the CSV; the table exists so the operational DB
    is not an empty schema next to a live decisions log.
    """
    from week1 import models
    from week1.database import SessionLocal, init_db

    init_db()
    session = SessionLocal()
    try:
        session.query(models.Shipment).delete()
        rows = [
            models.Shipment(
                shipment_date=str(rec["shipment_date"]) if rec.get("shipment_date") else None,
                sku=rec["sku"],
                supplier=rec["supplier"],
                origin_region=rec["origin_region"],
                distance_km=float(rec["distance_km"]),
                historical_avg_lead_time_days=float(rec["historical_avg_lead_time_days"]),
                order_quantity=int(rec["order_quantity"]),
                unit_cost_usd=float(rec["unit_cost_usd"]),
                is_peak_season=bool(rec["is_peak_season"]),
                actual_delay_days=float(rec["actual_delay_days"]),
            )
            for rec in df.to_dict(orient="records")
        ]
        session.bulk_save_objects(rows)
        session.commit()
        return len(rows)
    finally:
        session.close()


if __name__ == "__main__":
    main()
