"""
Week 1 — break shipments.csv into train / validation / test (60:20:20).

Temporal split when shipment_date exists (earlier → later).
Writes CSVs under data/ only — those files are gitignored so they are
never pushed to GitHub.

    python week1/generate_mock_data.py
    python week1/split_dataset.py

Outputs (local only, not committed):
    data/train.csv
    data/validation.csv
    data/test.csv
    data/dataset_split_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from week1.config import ROOT_DIR
from week1.delay_model import _temporal_split_masks

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"
OUT_DIR = ROOT_DIR / "data"
TRAIN_PATH = OUT_DIR / "train.csv"
VAL_PATH = OUT_DIR / "validation.csv"
TEST_PATH = OUT_DIR / "test.csv"
SUMMARY_PATH = OUT_DIR / "dataset_split_summary.json"

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    df = df.reset_index(drop=True)
    if "shipment_date" in df.columns:
        dates = pd.to_datetime(df["shipment_date"])
        train_mask, val_mask, test_mask = _temporal_split_masks(
            dates, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC
        )
        strategy = "temporal_60_20_20"
        return df.loc[train_mask], df.loc[val_mask], df.loc[test_mask], strategy

    # Random fallback matching DelayModel when dates are absent.
    from sklearn.model_selection import train_test_split

    temp, test = train_test_split(df, test_size=TEST_FRAC, random_state=13)
    train, val = train_test_split(temp, test_size=0.25, random_state=13)
    return train, val, test, "random_60_20_20_fallback"


def summarize_split(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, strategy: str
) -> dict:
    n = len(train) + len(val) + len(test)
    summary = {
        "heading": "Dataset analysis",
        "project": "Data Analytics by Axlero — SupplyPrescript",
        "split_ratio": "60:20:20",
        "split_labels": {
            "train": "Train",
            "validation": "Data validation",
            "test": "Testing",
        },
        "strategy": strategy,
        "n_total": n,
        "n_train": len(train),
        "n_validation": len(val),
        "n_test": len(test),
        "pct_train": round(100 * len(train) / n, 2) if n else 0,
        "pct_validation": round(100 * len(val) / n, 2) if n else 0,
        "pct_test": round(100 * len(test) / n, 2) if n else 0,
        "outputs": {
            "train": str(TRAIN_PATH.name),
            "validation": str(VAL_PATH.name),
            "test": str(TEST_PATH.name),
        },
        "note": (
            "Split CSV files live under data/ and are gitignored — "
            "they are not committed to GitHub."
        ),
    }
    if "shipment_date" in train.columns:
        for name, part in (("train", train), ("validation", val), ("test", test)):
            dates = pd.to_datetime(part["shipment_date"])
            summary[f"{name}_date_range"] = {
                "min": str(dates.min().date()),
                "max": str(dates.max().date()),
            }
    return summary


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found — run week1/generate_mock_data.py first")

    df = pd.read_csv(DATA_PATH)
    train, val, test, strategy = split_frame(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    summary = summarize_split(train, val, test, strategy)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("=== Dataset analysis ===")
    print("Data Analytics by Axlero — SupplyPrescript")
    print(f"Split: Train : Data validation : Testing = {TRAIN_FRAC:.0%}:{VAL_FRAC:.0%}:{TEST_FRAC:.0%}")
    print(f"Strategy: {strategy}")
    print(f"  Train            : {summary['n_train']:,} ({summary['pct_train']}%) -> {TRAIN_PATH.name}")
    print(f"  Data validation  : {summary['n_validation']:,} ({summary['pct_validation']}%) -> {VAL_PATH.name}")
    print(f"  Testing          : {summary['n_test']:,} ({summary['pct_test']}%) -> {TEST_PATH.name}")
    print()
    print("Local Windows copy target (optional):")
    print(r"  C:\Users\ACER\Desktop\Data Analytics by Axlero")
    print()
    print("Note: data/*.csv is gitignored — split files are not pushed to GitHub.")
    print(f"Summary written -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
