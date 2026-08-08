"""
XGBoost ML process + confusion-matrix evaluation (SupplyPrescript).

Heading: Dataset analysis — ML Process for XGBoost

Steps
-----
1. Load shipment history (or generate mock data)
2. Build features
3. Temporal 60:20:20 train / data validation / testing split
4. Fit XGBoost classifier (late?) + regressor (delay days)
5. Score the test set; compare to supplier baselines
6. Plot confusion matrix with TN / FP / FN / TP labels
7. Write a short verdict: is the model doing right?

Outputs are local only (gitignored under data/ml_evaluation/ and
exports/). Copy the ready-made folder to:

    C:\\Users\\ACER\\Desktop\\Data Analytics by Axlero

Run:
    python week1/generate_mock_data.py   # if needed
    python week1/evaluate_xgboost.py
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from week1.config import ROOT_DIR
from week1.delay_model import DelayModel
from week1.features import DELAY_FLAG_THRESHOLD_DAYS

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"
EVAL_DIR = ROOT_DIR / "data" / "ml_evaluation"
EXPORT_DIR = ROOT_DIR / "exports" / "Data Analytics by Axlero"
WINDOWS_COPY_PATH = r"C:\Users\ACER\Desktop\Data Analytics by Axlero"

CM_PNG = "xgboost_confusion_matrix.png"
CM_ANNOTATED_PNG = "xgboost_tp_fp_matrix.png"
PROCESS_MD = "ML Process for XGBoost.md"
SUMMARY_JSON = "xgboost_ml_evaluation.json"
ZIP_NAME = "Data_Analytics_by_Axlero_XGBoost.zip"


def _ensure_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from week1.generate_mock_data import build

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = build()
        df.to_csv(DATA_PATH, index=False)
        print(f"generated {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def _verdict(metrics: dict) -> dict:
    """Ruthless check: lift must be real, calibrated, and operationally sane.

    Soft bars (AUC ≥ baseline, TP > 0, FPR ≤ 50%) were too weak for a system
    that multiplies P(delay) into money. Failures here mean: do not trust the
    model for prescriptions until fixed.
    """
    auc = metrics.get("auc")
    baseline_auc = metrics.get("baseline_auc")
    mae = metrics.get("mae_days")
    baseline_mae = metrics.get("baseline_mae_days")
    precision = metrics.get("precision") or 0.0
    recall = metrics.get("recall") or 0.0
    f1 = metrics.get("f1") or 0.0
    brier = metrics.get("brier_score")
    ece = metrics.get("ece")
    cm = metrics.get("confusion_matrix") or [[0, 0], [0, 0]]
    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    n_pos = tp + fn
    n_neg = tn + fp

    checks = []
    ok = True
    warnings = []

    if auc is not None and baseline_auc is not None:
        beat_auc = auc >= baseline_auc + 0.02  # require meaningful lift, not noise
        checks.append(
            {
                "check": "Classifier AUC ≥ baseline + 0.02",
                "model": auc,
                "baseline": baseline_auc,
                "pass": beat_auc,
            }
        )
        ok = ok and beat_auc

    auc_boot = metrics.get("auc_lift_bootstrap") or {}
    if auc_boot.get("significant") is not None:
        sig = bool(auc_boot["significant"])
        checks.append(
            {
                "check": "AUC lift bootstrap 95% CI excludes 0",
                "model": auc_boot.get("lift"),
                "ci": [auc_boot.get("ci_low"), auc_boot.get("ci_high")],
                "pass": sig,
            }
        )
        ok = ok and sig

    if mae is not None and baseline_mae is not None:
        # Require at least ~3% relative MAE improvement vs supplier mean.
        relative = (baseline_mae - mae) / baseline_mae if baseline_mae else 0.0
        beat_mae = mae <= baseline_mae and relative >= 0.03
        checks.append(
            {
                "check": "Regressor MAE ≤ baseline with ≥3% relative lift",
                "model": mae,
                "baseline": baseline_mae,
                "relative_lift": round(relative, 4),
                "pass": beat_mae,
            }
        )
        ok = ok and beat_mae

    mae_boot = metrics.get("mae_lift_bootstrap") or {}
    if mae_boot.get("significant") is False:
        warnings.append(
            "MAE lift bootstrap CI crosses 0 — magnitude improvement may be noise."
        )
    elif mae_boot.get("significant") is True:
        checks.append(
            {
                "check": "MAE lift bootstrap 95% CI excludes 0",
                "model": mae_boot.get("lift"),
                "ci": [mae_boot.get("ci_low"), mae_boot.get("ci_high")],
                "pass": True,
            }
        )

    # Recall on late class: missing a late shipment is the expensive FN.
    recall_ok = recall >= 0.55 if n_pos else False
    checks.append(
        {
            "check": "Recall (late class) ≥ 0.55",
            "model": recall,
            "pass": recall_ok,
        }
    )
    ok = ok and recall_ok

    finds_late = tp >= max(5, int(0.05 * n_pos)) if n_pos else False
    checks.append(
        {
            "check": "True positives ≥ max(5, 5% of late shipments)",
            "model": tp,
            "pass": finds_late,
        }
    )
    ok = ok and finds_late

    false_alarm_rate = fp / (fp + tn) if (fp + tn) else None
    if false_alarm_rate is not None:
        fp_ok = false_alarm_rate <= 0.35
        checks.append(
            {
                "check": "False-positive rate among on-time ≤ 35%",
                "model": round(false_alarm_rate, 3),
                "pass": fp_ok,
            }
        )
        ok = ok and fp_ok

    if brier is not None:
        brier_ok = brier <= 0.22
        checks.append({"check": "Brier score ≤ 0.22", "model": brier, "pass": brier_ok})
        ok = ok and brier_ok

    if ece is not None:
        ece_ok = ece <= 0.12
        checks.append({"check": "Expected calibration error (ECE) ≤ 0.12", "model": ece, "pass": ece_ok})
        ok = ok and ece_ok

    if metrics.get("probability_calibrated") is False:
        warnings.append("Probabilities are NOT calibrated — unsafe for expected-cost decisions.")

    if metrics.get("clf_reg_consistency_ok") is False:
        warnings.append("Classifier/regressor disagree: predicted-late rows have low delay days.")
        ok = False

    weak = metrics.get("weak_segments_auc_lt_0_60") or {}
    if any(weak.values()):
        warnings.append(f"Weak segments (AUC<0.60): {weak}")

    return {
        "model_doing_right": ok,
        "summary": (
            "Model clears ruthless scientific gates vs baselines on the test hold-out."
            if ok
            else "Model FAILS one or more scientific gates — do not trust prescriptions yet."
        ),
        "checks": checks,
        "warnings": warnings,
        "confusion_counts": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "decision_threshold": metrics.get("decision_threshold"),
        "note": (
            "Metrics recover relationships programmed into synthetic data; "
            "they are not real-world supply-chain accuracy claims."
        ),
    }


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> Path:
    """Standard sklearn-style confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["On-time (0)", "Late (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("XGBoost — Confusion Matrix (test set)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_tp_fp_matrix(cm: np.ndarray, out_path: Path) -> Path:
    """Confusion matrix annotated with TN / FP / FN / TP names."""
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    labels = np.array(
        [
            [f"TN\nTrue Negative\n{tn}", f"FP\nFalse Positive\n{fp}"],
            [f"FN\nFalse Negative\n{fn}", f"TP\nTrue Positive\n{tp}"],
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], ["Pred On-time", "Pred Late"])
    ax.set_yticks([0, 1], ["True On-time", "True Late"])
    ax.set_title("XGBoost — True Positive / False Positive Matrix")

    threshold = cm.max() / 2.0 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "#1c2430"
            ax.text(j, i, labels[i, j], ha="center", va="center", color=color, fontsize=11)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_process_markdown(path: Path, metrics: dict, verdict: dict) -> Path:
    cm = verdict["confusion_counts"]
    lines = [
        "# Dataset analysis",
        "",
        "## ML Process for XGBoost",
        "",
        "**Data Analytics by Axlero — SupplyPrescript**",
        "",
        f"Copy this folder to: `{WINDOWS_COPY_PATH}`",
        "",
        "### Process steps",
        "",
        "1. **Data** — synthetic shipment history with `shipment_date`",
        "2. **Features** — supplier / region / SKU one-hots + distance, lead time, quantity, cost, peak season",
        "3. **Split** — Train : Data validation : Testing = **60 : 20 : 20** (temporal)",
        "4. **Models** — `XGBClassifier` (late > 3 days?) + `XGBRegressor` (delay days)",
        "5. **Evaluate** — test-set AUC / MAE vs supplier baselines; confusion matrix",
        "6. **Decide** — is the model doing right?",
        "",
        "### Is the model doing right?",
        "",
        f"**Verdict:** {verdict['summary']}",
        "",
        f"- Model doing right: **{verdict['model_doing_right']}**",
        f"- Validation strategy: `{metrics.get('validation_strategy')}`",
        f"- Train / Val / Test: {metrics.get('n_train')} / {metrics.get('n_val')} / {metrics.get('n_test')}",
        f"- Classifier AUC / PR-AUC: {metrics.get('auc')} / {metrics.get('pr_auc')} "
        f"(baseline AUC {metrics.get('baseline_auc')})",
        f"- Regressor MAE / RMSE / R²: {metrics.get('mae_days')} / {metrics.get('rmse_days')} / "
        f"{metrics.get('r2_days')} (baseline MAE {metrics.get('baseline_mae_days')})",
        f"- Precision / Recall / F1 / Specificity: {metrics.get('precision')} / "
        f"{metrics.get('recall')} / {metrics.get('f1')} / {metrics.get('specificity')}",
        f"- Brier / ECE: {metrics.get('brier_score')} / {metrics.get('ece')}",
        f"- Decision threshold (val-tuned): {metrics.get('decision_threshold')}",
        f"- Probability calibrated: {metrics.get('probability_calibrated')}",
        f"- AUC lift bootstrap: {metrics.get('auc_lift_bootstrap')}",
        f"- MAE lift bootstrap: {metrics.get('mae_lift_bootstrap')}",
        "",
        "### Ruthless gates",
        "",
    ]
    for check in verdict.get("checks", []):
        mark = "PASS" if check.get("pass") else "FAIL"
        lines.append(f"- [{mark}] {check.get('check')}: model={check.get('model')}")
    if verdict.get("warnings"):
        lines.append("")
        lines.append("### Warnings")
        lines.append("")
        for w in verdict["warnings"]:
            lines.append(f"- {w}")
    lines.extend(
        [
            "",
            "### Confusion matrix counts (test set)",
            "",
            "| Cell | Meaning | Count |",
            "|---|---|---|",
            f"| **TN** | True Negative — correctly predicted on-time | {cm['true_negative']} |",
            f"| **FP** | False Positive — predicted late, actually on-time | {cm['false_positive']} |",
            f"| **FN** | False Negative — predicted on-time, actually late | {cm['false_negative']} |",
            f"| **TP** | True Positive — correctly predicted late | {cm['true_positive']} |",
            "",
            "### Files in this folder",
            "",
            f"- `{CM_PNG}` — standard confusion matrix plot",
            f"- `{CM_ANNOTATED_PNG}` — TP / FP / TN / FN annotated matrix",
            f"- `{SUMMARY_JSON}` — metrics + verdict (machine-readable)",
            f"- `{PROCESS_MD}` — this document",
            "",
            "### Note",
            "",
            verdict["note"],
            "",
            "Generated plots and CSVs are **gitignored** and are not committed to GitHub.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _copy_tree(src_files: list[Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for src in src_files:
        shutil.copy2(src, dest / src.name)


def _make_zip(folder: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path("Data Analytics by Axlero") / path.relative_to(folder)))
    return zip_path


def main() -> dict:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df = _ensure_data()

    model = DelayModel()
    metrics = model.fit(df, verbose=True)
    cm_list = metrics["confusion_matrix"]
    cm = np.array(cm_list, dtype=int)
    # Rebuild explicit TN/FP/FN/TP from the same test predictions for the plot
    # (already in metrics['confusion_matrix'] from DelayModel).
    assert cm.shape == (2, 2)

    verdict = _verdict(metrics)

    cm_path = plot_confusion_matrix(cm, EVAL_DIR / CM_PNG)
    tp_fp_path = plot_tp_fp_matrix(cm, EVAL_DIR / CM_ANNOTATED_PNG)

    payload = {
        "heading": "Dataset analysis",
        "title": "ML Process for XGBoost",
        "project": "Data Analytics by Axlero — SupplyPrescript",
        "windows_copy_path": WINDOWS_COPY_PATH,
        "delay_flag_threshold_days": DELAY_FLAG_THRESHOLD_DAYS,
        "metrics": {
            k: metrics[k]
            for k in (
                "validation_strategy",
                "validation_used_for",
                "n_train",
                "n_val",
                "n_test",
                "auc",
                "pr_auc",
                "baseline_auc",
                "auc_lift_vs_baseline",
                "auc_lift_bootstrap",
                "mae_days",
                "rmse_days",
                "r2_days",
                "baseline_mae_days",
                "mae_lift_vs_baseline",
                "mae_lift_bootstrap",
                "precision",
                "recall",
                "f1",
                "specificity",
                "npv",
                "false_alarm_rate",
                "brier_score",
                "ece",
                "decision_threshold",
                "probability_calibrated",
                "confusion_matrix",
                "weak_segments_auc_lt_0_60",
                "clf_reg_consistency_ok",
                "data_is_synthetic",
            )
            if k in metrics
        },
        "verdict": verdict,
        "plots": [CM_PNG, CM_ANNOTATED_PNG],
        "note": "Evaluation outputs are local/gitignored — not pushed to GitHub.",
    }
    summary_path = EVAL_DIR / SUMMARY_JSON
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = write_process_markdown(EVAL_DIR / PROCESS_MD, metrics, verdict)

    # Ready-to-copy package matching the Windows Desktop folder name
    package_files = [cm_path, tp_fp_path, summary_path, md_path]
    _copy_tree(package_files, EXPORT_DIR)
    zip_path = _make_zip(EXPORT_DIR, ROOT_DIR / "exports" / ZIP_NAME)

    print()
    print("=== Dataset analysis — ML Process for XGBoost ===")
    print(verdict["summary"])
    print(
        f"Confusion matrix [[TN, FP], [FN, TP]] = "
        f"{verdict['confusion_counts']}"
    )
    print(f"Plots: {cm_path.name}, {tp_fp_path.name}")
    print()
    print("Copy-ready folder:")
    print(f"  {EXPORT_DIR}")
    print(f"Zip (extract into Desktop):")
    print(f"  {zip_path}")
    print(f"Target on your PC:")
    print(f"  {WINDOWS_COPY_PATH}")
    print()
    print("Note: evaluation files are gitignored — not committed to GitHub.")
    return payload


if __name__ == "__main__":
    main()
