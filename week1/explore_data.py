"""
Week 1 — exploratory data analysis (EDA).

Run this after generate_mock_data.py to produce the charts a hiring
manager expects to see in a data-analytics portfolio:

    python week1/explore_data.py

Outputs PNGs under docs/figures/ and prints a short summary table.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from week1.config import ROOT_DIR

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"
FIG_DIR = ROOT_DIR / "docs" / "figures"


def load() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found — run week1/generate_mock_data.py first")
    return pd.read_csv(DATA_PATH)


def summarize(df: pd.DataFrame) -> None:
    print("=== Shipment history snapshot ===")
    print(f"rows: {len(df):,}")
    print(f"suppliers: {df['supplier'].nunique()}")
    print(f"skus: {df['sku'].nunique()}")
    if "shipment_date" in df.columns:
        print(f"date range: {df['shipment_date'].min()} .. {df['shipment_date'].max()}")
    print()
    print("Delay days by supplier (mean / p90):")
    by_supplier = df.groupby("supplier")["actual_delay_days"].agg(["mean", lambda s: s.quantile(0.9)])
    by_supplier.columns = ["mean_days", "p90_days"]
    print(by_supplier.round(2).sort_values("mean_days", ascending=False).to_string())
    print()
    late_rate = (df["actual_delay_days"] > 3).mean()
    print(f"share of shipments late > 3 days: {late_rate:.1%}")
    peak = df.groupby("is_peak_season")["actual_delay_days"].mean()
    print(f"mean delay off-peak vs peak: {peak.get(False, float('nan')):.2f}d / {peak.get(True, float('nan')):.2f}d")
    if "shipment_date" in df.columns:
        delta = df[df["supplier"] == "Delta Cove Electronics"].copy()
        delta["shipment_date"] = pd.to_datetime(delta["shipment_date"])
        bad = delta[
            (delta["shipment_date"] >= "2024-07-01") & (delta["shipment_date"] <= "2024-09-30")
        ]
        other = delta[
            ~((delta["shipment_date"] >= "2024-07-01") & (delta["shipment_date"] <= "2024-09-30"))
        ]
        if len(bad) and len(other):
            print(
                f"Delta Cove mean delay bad-quarter vs other: "
                f"{bad['actual_delay_days'].mean():.2f}d / {other['actual_delay_days'].mean():.2f}d"
            )


def plot_delay_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["actual_delay_days"], bins=40, color="#2f6f5e", edgecolor="white")
    ax.axvline(3, color="#a4462f", linestyle="--", label="late threshold (3 days)")
    ax.set_xlabel("Actual delay (days)")
    ax.set_ylabel("Shipments")
    ax.set_title("Delay distribution — SupplyPrescript synthetic history")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "delay_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_delay_by_supplier(df: pd.DataFrame) -> Path:
    means = (
        df.groupby("supplier")["actual_delay_days"]
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(means.index, means.values, color="#1c2430")
    ax.set_xlabel("Mean delay (days)")
    ax.set_title("Mean delay by supplier")
    fig.tight_layout()
    out = FIG_DIR / "delay_by_supplier.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_peak_effect(df: pd.DataFrame) -> Path:
    pivot = (
        df.assign(season=df["is_peak_season"].map({True: "Peak (Nov–Dec)", False: "Off-peak"}))
        .groupby(["origin_region", "season"])["actual_delay_days"]
        .mean()
        .unstack("season")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot(kind="bar", ax=ax, color=["#5b6472", "#2f6f5e"], rot=0)
    ax.set_ylabel("Mean delay (days)")
    ax.set_xlabel("")
    ax.set_title("Peak-season effect by origin region")
    ax.legend(title="")
    fig.tight_layout()
    out = FIG_DIR / "peak_season_effect.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_delta_cove_bad_quarter(df: pd.DataFrame) -> Path | None:
    """Show calendar-bounded supplier deterioration (not a row-index shock)."""
    if "shipment_date" not in df.columns:
        return None
    delta = df[df["supplier"] == "Delta Cove Electronics"].copy()
    if delta.empty:
        return None
    delta["shipment_date"] = pd.to_datetime(delta["shipment_date"])
    delta = delta.sort_values("shipment_date")
    monthly = delta.set_index("shipment_date")["actual_delay_days"].resample("MS").mean()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(monthly.index, monthly.values, color="#1c2430", marker="o", markersize=3)
    ax.axvspan(pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30"), color="#a4462f", alpha=0.2, label="Bad quarter")
    ax.set_ylabel("Mean delay (days)")
    ax.set_title("Delta Cove — monthly mean delay (calendar bad quarter)")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "delta_cove_bad_quarter.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    summarize(df)
    paths = [
        plot_delay_distribution(df),
        plot_delay_by_supplier(df),
        plot_peak_effect(df),
    ]
    bad_q = plot_delta_cove_bad_quarter(df)
    if bad_q is not None:
        paths.append(bad_q)
    print()
    print("figures written:")
    for path in paths:
        print(f"  - {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
