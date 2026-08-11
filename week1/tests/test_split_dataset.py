"""Tests for the 60:20:20 dataset split helper."""
from __future__ import annotations

import json

import pandas as pd

from week1.generate_mock_data import build
from week1 import split_dataset as split_mod
from week1.split_dataset import split_frame


def test_split_frame_is_approximately_60_20_20():
    df = build(n_rows=1000)
    train, val, test, strategy = split_frame(df)
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert strategy == "temporal_60_20_20"
    assert abs(len(train) / total - 0.60) < 0.02
    assert abs(len(val) / total - 0.20) < 0.02
    assert abs(len(test) / total - 0.20) < 0.02


def test_temporal_split_preserves_time_order():
    df = build(n_rows=800)
    train, val, test, _ = split_frame(df)
    train_max = pd.to_datetime(train["shipment_date"]).max()
    val_min = pd.to_datetime(val["shipment_date"]).min()
    val_max = pd.to_datetime(val["shipment_date"]).max()
    test_min = pd.to_datetime(test["shipment_date"]).min()
    assert train_max <= val_min
    assert val_max <= test_min


def test_main_writes_local_csvs_only(tmp_path, monkeypatch):
    """Split artifacts go under a temp data dir (mirrors gitignored data/)."""
    df = build(n_rows=500)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shipments = data_dir / "shipments.csv"
    df.to_csv(shipments, index=False)

    monkeypatch.setattr(split_mod, "DATA_PATH", shipments)
    monkeypatch.setattr(split_mod, "OUT_DIR", data_dir)
    monkeypatch.setattr(split_mod, "TRAIN_PATH", data_dir / "train.csv")
    monkeypatch.setattr(split_mod, "VAL_PATH", data_dir / "validation.csv")
    monkeypatch.setattr(split_mod, "TEST_PATH", data_dir / "test.csv")
    monkeypatch.setattr(split_mod, "SUMMARY_PATH", data_dir / "dataset_split_summary.json")

    split_mod.main()

    assert (data_dir / "train.csv").exists()
    assert (data_dir / "validation.csv").exists()
    assert (data_dir / "test.csv").exists()
    summary = json.loads((data_dir / "dataset_split_summary.json").read_text())
    assert summary["heading"] == "Dataset analysis"
    assert summary["split_ratio"] == "60:20:20"
    assert summary["n_train"] + summary["n_validation"] + summary["n_test"] == len(df)
