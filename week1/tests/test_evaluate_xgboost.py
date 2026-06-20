"""Tests for XGBoost ML evaluation + confusion-matrix plots."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from week1 import evaluate_xgboost as ev
from week1.generate_mock_data import build


def test_verdict_passes_when_model_beats_baseline():
    metrics = {
        "auc": 0.85,
        "baseline_auc": 0.70,
        "mae_days": 1.5,
        "baseline_mae_days": 2.0,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
        "confusion_matrix": [[80, 10], [5, 40]],
    }
    verdict = ev._verdict(metrics)
    assert verdict["model_doing_right"] is True
    assert verdict["confusion_counts"]["true_positive"] == 40
    assert verdict["confusion_counts"]["false_positive"] == 10


def test_verdict_fails_when_auc_below_baseline():
    metrics = {
        "auc": 0.55,
        "baseline_auc": 0.70,
        "mae_days": 1.5,
        "baseline_mae_days": 2.0,
        "precision": 0.5,
        "recall": 0.4,
        "f1": 0.45,
        "confusion_matrix": [[50, 40], [30, 20]],
    }
    verdict = ev._verdict(metrics)
    assert verdict["model_doing_right"] is False


def test_plot_confusion_matrix_writes_png(tmp_path):
    cm = np.array([[90, 8], [12, 40]], dtype=int)
    out = tmp_path / "cm.png"
    path = ev.plot_confusion_matrix(cm, out)
    assert path.exists() and path.stat().st_size > 0


def test_plot_tp_fp_matrix_writes_png(tmp_path):
    cm = np.array([[90, 8], [12, 40]], dtype=int)
    out = tmp_path / "tp_fp.png"
    path = ev.plot_tp_fp_matrix(cm, out)
    assert path.exists() and path.stat().st_size > 0


def test_main_writes_gitignored_exports(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    eval_dir = data_dir / "ml_evaluation"
    export_dir = tmp_path / "exports" / "Data Analytics by Axlero"
    exports_root = tmp_path / "exports"
    data_dir.mkdir(parents=True)
    df = build(n_rows=600)
    shipments = data_dir / "shipments.csv"
    df.to_csv(shipments, index=False)

    monkeypatch.setattr(ev, "DATA_PATH", shipments)
    monkeypatch.setattr(ev, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(ev, "EXPORT_DIR", export_dir)
    monkeypatch.setattr(ev, "ROOT_DIR", tmp_path)

    # Point zip into tmp exports
    payload = ev.main()
    assert payload["verdict"]["confusion_counts"]["true_positive"] >= 0
    assert (eval_dir / ev.CM_PNG).exists()
    assert (eval_dir / ev.CM_ANNOTATED_PNG).exists()
    assert (eval_dir / ev.SUMMARY_JSON).exists()
    assert (export_dir / ev.CM_PNG).exists()
    assert (export_dir / ev.PROCESS_MD).exists()
    summary = json.loads((eval_dir / ev.SUMMARY_JSON).read_text())
    assert summary["heading"] == "Dataset analysis"
    assert summary["title"] == "ML Process for XGBoost"
    zip_path = exports_root / ev.ZIP_NAME
    assert zip_path.exists()
