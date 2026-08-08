"""
Week 1 — ingest real open shipment data into CSV + the SQLAlchemy DB.

Replaces the ~4,000-row mock generator as the default data path:

    python week1/ingest_real_data.py                 # USAID SCMS (default)
    python week1/ingest_real_data.py --source uci-c2k
    python week1/ingest_real_data.py --source both
    python week1/ingest_real_data.py --force-download

Writes:
  - data/raw/…              cached source extracts
  - data/shipments.csv      training extract (same schema as before)
  - data/supplyprescript.db shipments table (or DATABASE_URL target)

Then train as usual:  python week1/train_model.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from week1.config import ROOT_DIR
from week1.data_adapters import ADAPTERS, SHIPMENT_COLUMNS
from week1.database import SessionLocal, init_db
from week1.models import Shipment

RAW_DIR = ROOT_DIR / "data" / "raw"
CSV_PATH = ROOT_DIR / "data" / "shipments.csv"


def load_source(source: str, force_download: bool = False) -> pd.DataFrame:
    meta = ADAPTERS[source]
    cached = __import__("week1.data_adapters", fromlist=["cache_raw"]).cache_raw(
        RAW_DIR, meta["cache_name"], meta["urls"], force=force_download
    )
    print(f"[{source}] using {cached} ({cached.stat().st_size:,} bytes)")
    df = meta["transform"](cached)
    df["data_source"] = source
    print(f"[{source}] mapped {len(df):,} rows from {meta['label']}")
    return df


def seed_database(df: pd.DataFrame, replace: bool = True) -> int:
    """Persist shipment history into the configured database."""
    init_db()
    session = SessionLocal()
    try:
        if replace:
            deleted = session.query(Shipment).delete()
            session.commit()
            print(f"[db] cleared {deleted} existing shipment rows")

        # SQLAlchemy bulk insert via ORM objects (portable across sqlite/postgres).
        has_source = "data_source" in Shipment.__table__.columns.keys()
        batch: list[Shipment] = []
        for record in df.to_dict(orient="records"):
            payload = {col: record[col] for col in SHIPMENT_COLUMNS}
            if has_source:
                payload["data_source"] = record.get("data_source")
            batch.append(Shipment(**payload))
            if len(batch) >= 500:
                session.add_all(batch)
                session.commit()
                batch.clear()
        if batch:
            session.add_all(batch)
            session.commit()
        count = session.query(Shipment).count()
        print(f"[db] shipments table now has {count:,} rows")
        return count
    finally:
        session.close()


def shipments_from_db() -> pd.DataFrame | None:
    """Load training frame from DB when the shipments table is populated."""
    init_db()
    session = SessionLocal()
    try:
        count = session.query(Shipment).count()
        if count == 0:
            return None
        rows = session.query(Shipment).all()
        records = [
            {
                "sku": r.sku,
                "supplier": r.supplier,
                "origin_region": r.origin_region,
                "distance_km": r.distance_km,
                "historical_avg_lead_time_days": r.historical_avg_lead_time_days,
                "order_quantity": r.order_quantity,
                "unit_cost_usd": r.unit_cost_usd,
                "is_peak_season": bool(r.is_peak_season),
                "actual_delay_days": r.actual_delay_days,
            }
            for r in rows
        ]
        return pd.DataFrame(records, columns=SHIPMENT_COLUMNS)
    finally:
        session.close()


def load_shipments(prefer_db: bool = True) -> pd.DataFrame:
    """Shared loader for train / explore / retrain."""
    if prefer_db:
        db_df = shipments_from_db()
        if db_df is not None and len(db_df) > 0:
            return db_df
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    raise FileNotFoundError(
        f"No shipment data found. Run: python week1/ingest_real_data.py\n"
        f"(looked for DB shipments table and {CSV_PATH})"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest real open shipment datasets")
    parser.add_argument(
        "--source",
        choices=["usaid-scms", "uci-c2k", "both"],
        default="usaid-scms",
        help="usaid-scms ≈10k real SCMS rows (default); uci-c2k ≈3.9k UCI freight rows; both concatenates",
    )
    parser.add_argument("--force-download", action="store_true", help="Ignore data/raw cache")
    parser.add_argument("--skip-db", action="store_true", help="Only write CSV, do not seed DB")
    args = parser.parse_args(argv)

    sources = ["usaid-scms", "uci-c2k"] if args.source == "both" else [args.source]
    frames = [load_source(src, force_download=args.force_download) for src in sources]
    df = pd.concat(frames, ignore_index=True)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_cols = SHIPMENT_COLUMNS  # keep CSV schema stable for notebooks
    df[export_cols].to_csv(CSV_PATH, index=False)
    print(f"wrote {len(df):,} rows -> {CSV_PATH}")
    print(df["actual_delay_days"].describe().to_string())
    print()
    print("suppliers:", df["supplier"].nunique(), "| skus:", df["sku"].nunique(),
          "| regions:", sorted(df["origin_region"].unique()))
    late = (df["actual_delay_days"] > 3).mean()
    print(f"share late > 3 days: {late:.1%}")

    if not args.skip_db:
        seed_database(df, replace=True)


if __name__ == "__main__":
    main()
