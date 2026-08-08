"""
Week 1 — exploratory data analysis (EDA).

Run this after ingest_real_data.py to produce the charts a hiring
manager expects to see in a data-analytics portfolio:

    python week1/ingest_real_data.py
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
from week1.ingest_real_data import load_shipments

FIG_DIR = ROOT_DIR / "docs" / "figures"


def load() -> pd.DataFrame:
    try:
        return load_shipments(prefer_db=True)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


def summarize(df: pd.DataFrame) -> None:
    print("=== Shipment history snapshot ===")
    print(f"rows: {len(df):,}")
    print(f"suppliers: {df['supplier'].nunique()}")
    print(f"skus: {df['sku'].nunique()}")
    print()
    print("Delay days by supplier (mean / p90) — top 12 by mean delay:")
    by_supplier = (
        df.groupby("supplier")["actual_delay_days"]
        .agg(["mean", "count", lambda s: s.quantile(0.9)])
    )
    by_supplier.columns = ["mean_days", "n", "p90_days"]
    by_supplier = by_supplier.query("n >= 20").sort_values("mean_days", ascending=False).head(12)
    print(by_supplier.round(2).to_string())
    print()
    late_rate = (df["actual_delay_days"] > 3).mean()
    print(f"share of shipments late > 3 days: {late_rate:.1%}")
    peak = df.groupby("is_peak_season")["actual_delay_days"].mean()
    print(f"mean delay off-peak vs peak: {peak.get(False, float('nan')):.2f}d / {peak.get(True, float('nan')):.2f}d")


def plot_delay_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["actual_delay_days"], bins=40, color="#2f6f5e", edgecolor="white")
    ax.axvline(3, color="#a4462f", linestyle="--", label="late threshold (3 days)")
    ax.set_xlabel("Actual delay (days)")
    ax.set_ylabel("Shipments")
    ax.set_title("Delay distribution — SupplyPrescript shipment history")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "delay_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_delay_by_supplier(df: pd.DataFrame) -> Path:
    means = (
        df.groupby("supplier")["actual_delay_days"]
        .agg(["mean", "count"])
        .query("count >= 30")
        .sort_values("mean", ascending=True)
        .tail(15)["mean"]
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(means.index, means.values, color="#1c2430")
    ax.set_xlabel("Mean delay (days)")
    ax.set_title("Mean delay by supplier (top 15 by delay, n≥30)")
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


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    summarize(df)
    paths = [
        plot_delay_distribution(df),
        plot_delay_by_supplier(df),
        plot_peak_effect(df),
    ]
    print()
    print("figures written:")
    for path in paths:
        print(f"  - {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
