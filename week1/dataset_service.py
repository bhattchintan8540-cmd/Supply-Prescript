"""
Helpers that expose the real training extract (USAID SCMS / UCI C2K)
to the API and dashboard — summary stats, demo scenarios, and sample rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from week1.config import ROOT_DIR
from week1.ingest_real_data import load_shipments

DATASETS_DIR = ROOT_DIR / "datasets"
SOURCE_LABELS = {
    "usaid-scms": "USAID SCMS Delivery History",
    "uci-c2k": "UCI Cargo 2000 Freight Tracking",
    "mock": "Offline mock (fallback)",
}


def _safe_load() -> pd.DataFrame | None:
    try:
        return load_shipments(prefer_db=True)
    except FileNotFoundError:
        return None


def dataset_summary() -> dict:
    df = _safe_load()
    if df is None or df.empty:
        return {
            "available": False,
            "message": "No shipment data loaded. Run: python week1/ingest_real_data.py",
            "n_rows": 0,
            "n_suppliers": 0,
            "n_skus": 0,
            "n_regions": 0,
            "late_rate_pct": None,
            "mean_delay_days": None,
            "median_delay_days": None,
            "regions": [],
            "top_suppliers": [],
            "sources": [],
            "files": [],
        }

    late_rate = float((df["actual_delay_days"] > 3).mean() * 100)
    by_supplier = (
        df.groupby("supplier")["actual_delay_days"]
        .agg(["count", "mean"])
        .query("count >= 20")
        .sort_values("mean", ascending=False)
    )
    top = [
        {
            "supplier": idx,
            "n": int(row["count"]),
            "mean_delay_days": round(float(row["mean"]), 2),
        }
        for idx, row in by_supplier.head(8).iterrows()
    ]

    sources: list[dict] = []
    # Prefer provenance from separate dataset files when present.
    file_specs = [
        ("usaid-scms", DATASETS_DIR / "usaid_scms_shipments.csv", SOURCE_LABELS["usaid-scms"]),
        ("uci-c2k", DATASETS_DIR / "uci_c2k_shipments.csv", SOURCE_LABELS["uci-c2k"]),
        ("combined", DATASETS_DIR / "combined_shipments.csv", "Combined SCMS + C2K"),
    ]
    files = []
    for key, path, label in file_specs:
        if path.exists():
            n = max(sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1, 0)
            files.append({"key": key, "label": label, "path": str(path.relative_to(ROOT_DIR)), "n_rows": n})

    # Infer which package is currently loaded from the training row count.
    n_train = int(len(df))
    for f in files:
        if f["key"] != "combined" and abs(f["n_rows"] - n_train) <= 5:
            sources.append({"key": f["key"], "label": f["label"], "n_rows": n_train})
            break
    if not sources and files:
        # Combined or custom extract
        match = next((f for f in files if f["key"] == "combined" and abs(f["n_rows"] - n_train) <= 5), None)
        if match:
            sources.append({"key": "combined", "label": match["label"], "n_rows": n_train})
    if not sources:
        sources = [{"key": "training-extract", "label": "Training extract (CSV/DB)", "n_rows": n_train}]

    return {
        "available": True,
        "message": f"Real open shipment history loaded ({len(df):,} training rows)",
        "n_rows": int(len(df)),
        "n_suppliers": int(df["supplier"].nunique()),
        "n_skus": int(df["sku"].nunique()),
        "n_regions": int(df["origin_region"].nunique()),
        "late_rate_pct": round(late_rate, 1),
        "mean_delay_days": round(float(df["actual_delay_days"].mean()), 2),
        "median_delay_days": round(float(df["actual_delay_days"].median()), 2),
        "regions": sorted(df["origin_region"].dropna().unique().tolist()),
        "top_suppliers": top,
        "sources": sources,
        "files": files,
    }


def _row_to_shipment(row: pd.Series) -> dict:
    return {
        "sku": str(row["sku"]),
        "supplier": str(row["supplier"]),
        "origin_region": str(row["origin_region"]),
        "distance_km": round(float(row["distance_km"]), 1),
        "historical_avg_lead_time_days": round(float(row["historical_avg_lead_time_days"]), 1),
        "order_quantity": int(row["order_quantity"]),
        "unit_cost_usd": round(float(row["unit_cost_usd"]), 4),
        "is_peak_season": bool(row["is_peak_season"]),
    }


def demo_scenarios_from_data() -> list[dict]:
    """Build Demo A/B/C from real suppliers in the training extract."""
    df = _safe_load()
    if df is None or df.empty:
        return []

    stats = (
        df.groupby("supplier")
        .agg(
            n=("actual_delay_days", "size"),
            mean_delay=("actual_delay_days", "mean"),
            late_rate=("actual_delay_days", lambda s: float((s > 3).mean())),
        )
        .query("n >= 40")
        .sort_values("mean_delay")
    )
    if stats.empty:
        return []

    low_supplier = stats.index[0]
    high_supplier = stats.index[-1]

    def pick_row(supplier: str, prefer_peak: bool | None = None, prefer_late: bool = False) -> pd.Series:
        subset = df[df["supplier"] == supplier].copy()
        if prefer_late:
            late = subset[subset["actual_delay_days"] > 3]
            if not late.empty:
                subset = late
        if prefer_peak is True:
            peaked = subset[subset["is_peak_season"] == True]  # noqa: E712
            if not peaked.empty:
                subset = peaked
        elif prefer_peak is False:
            off = subset[subset["is_peak_season"] == False]  # noqa: E712
            if not off.empty:
                subset = off
        if prefer_late:
            # Among late shipments, take one near the upper quartile of delay.
            target = float(subset["actual_delay_days"].quantile(0.75))
            subset = subset.assign(_gap=(subset["actual_delay_days"] - target).abs())
            return subset.sort_values(["_gap", "order_quantity"], ascending=[True, False]).iloc[0]
        # Prefer a mid / typical order for the low-risk story.
        target = float(subset["actual_delay_days"].median())
        subset = subset.assign(_gap=(subset["actual_delay_days"] - target).abs())
        return subset.sort_values(["_gap", "order_quantity"]).iloc[len(subset) // 2]

    low_off = pick_row(low_supplier, prefer_peak=False, prefer_late=False)
    low_peak = pick_row(low_supplier, prefer_peak=True, prefer_late=False)
    high = pick_row(high_supplier, prefer_peak=True, prefer_late=True)

    low_ship = _row_to_shipment(low_off)
    peak_ship = _row_to_shipment(low_peak)
    peak_ship["is_peak_season"] = True  # force the contrast even if few peak rows
    high_ship = _row_to_shipment(high)
    high_ship["is_peak_season"] = True

    return [
        {
            "id": "safe",
            "label": f"Demo A · {low_supplier[:28]}",
            "blurb": f"Low historical delay ({stats.loc[low_supplier, 'mean_delay']:.1f}d mean)",
            "values": {
                **low_ship,
                "budget_cap_usd": 45000,
                "max_acceptable_delay_days": 5,
            },
        },
        {
            "id": "peak",
            "label": "Demo B · Same supplier / peak",
            "blurb": "Only seasonality changes",
            "values": {
                **peak_ship,
                "budget_cap_usd": 45000,
                "max_acceptable_delay_days": 5,
            },
        },
        {
            "id": "risky",
            "label": f"Demo C · {high_supplier[:28]}",
            "blurb": f"Higher delay risk ({stats.loc[high_supplier, 'mean_delay']:.1f}d mean)",
            "values": {
                **high_ship,
                "budget_cap_usd": 95000,
                "max_acceptable_delay_days": 5,
            },
        },
    ]


def sample_shipments(limit: int = 12) -> list[dict]:
    """Return a mix of low / high delay rows for the dashboard table."""
    df = _safe_load()
    if df is None or df.empty:
        return []

    high = df.nlargest(max(limit // 2, 1), "actual_delay_days")
    low = df.nsmallest(max(limit - len(high), 1), "actual_delay_days")
    sample = pd.concat([high, low]).drop_duplicates().head(limit)
    rows = []
    for _, row in sample.iterrows():
        item = _row_to_shipment(row)
        item["actual_delay_days"] = round(float(row["actual_delay_days"]), 1)
        rows.append(item)
    return rows
