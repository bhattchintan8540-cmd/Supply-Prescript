"""
Week 1 - trains the delay model against data/shipments.csv and drops the
artifact at the path in week1/config.MODEL_PATH.

    python week1/train_model.py

Also called from week4/retrain.py once decisions start piling up and
drift crosses the threshold - that's the "continuous learning" piece
from Week 4.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from week1.config import MODEL_PATH, ROOT_DIR
from week1.delay_model import DelayModel

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found - run generate_mock_data.py first")

    df = pd.read_csv(DATA_PATH)
    model = DelayModel()
    metrics = model.fit(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"saved model -> {MODEL_PATH}")
    print(metrics)


if __name__ == "__main__":
    main()
