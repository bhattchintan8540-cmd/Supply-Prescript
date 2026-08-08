"""
Week 1 - trains the delay model against the real shipments table
(preferred) or data/shipments.csv, and drops the artifact at the path
in week1/config.MODEL_PATH.

    python week1/ingest_real_data.py   # once — pull USAID SCMS / UCI C2K
    python week1/train_model.py

Also called from week4/retrain.py once decisions start piling up and
drift crosses the threshold - that's the "continuous learning" piece
from Week 4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week1.config import MODEL_PATH, ROOT_DIR
from week1.delay_model import DelayModel
from week1.ingest_real_data import load_shipments

METRICS_PATH = ROOT_DIR / "data" / "metrics.json"


def main() -> None:
    try:
        df = load_shipments(prefer_db=True)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"training on {len(df):,} shipment rows")
    model = DelayModel()
    metrics = model.fit(df)

    # Convert numpy scalars so metrics.json is plain JSON.
    clean = {k: (float(v) if v is not None and hasattr(v, "item") else v) for k, v in metrics.items()}
    for key, value in list(clean.items()):
        if isinstance(value, (int, float)) and key.startswith("n_"):
            clean[key] = int(value)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    importance = model.feature_importance(top_n=10)
    clean["top_features"] = importance
    METRICS_PATH.write_text(json.dumps(clean, indent=2))
    print(f"saved model -> {MODEL_PATH}")
    print(f"saved metrics -> {METRICS_PATH}")
    print(clean)
    print("top features (regressor):")
    for row in importance:
        print(f"  {row['importance']:.4f}  {row['feature']}")


if __name__ == "__main__":
    main()
