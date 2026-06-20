# Week 1 — Predictive Baseline + App Scaffolding

**Implementation Phase 1** of SupplyPrescript: everything needed before
there's anything to prescribe.

- `generate_mock_data.py` — builds a plausible 3-year shipment history
  with **shipment dates** and a true calendar bad quarter for Delta Cove
  (`../data/shipments.csv`). Relationships are programmed into the
  synthetic environment on purpose.
- `explore_data.py` — beginner-friendly EDA: summary tables + charts in
  `docs/figures/`. Pair with `notebooks/01_exploratory_analysis.ipynb`.
- `features.py` — turns a shipment row into a numeric feature vector.
  Shared by both training and live inference so the two can't quietly
  drift apart from each other.
- `delay_model.py` — XGBoost classifier + regressor with **temporal**
  60:20:20 train/val/test when dates exist, supplier baselines, and classifier
  diagnostics (precision/recall/F1/Brier/segments).
- `split_dataset.py` — export the same 60:20:20 partitions for analysis.
- `evaluate_xgboost.py` — ML-process evaluation with confusion-matrix plots
  and a baseline verdict (Week 5 packaging).
- `train_model.py` — CLI entry point: trains against the CSV, saves the
  artifact to `../data/delay_model.joblib` and metrics to
  `../data/metrics.json`.
- `demo_model.py` — **presentation demo**: compares four shipments so
  the audience can see supplier + peak-season effects live.
- `config.py` / `database.py` / `models.py` — scaffolding later weeks
  build on (`Shipment`, `Decision` with no-action cost + feature snapshot).

## Run it

```bash
python week1/generate_mock_data.py
python week1/explore_data.py
python week1/train_model.py
python week1/demo_model.py          # live model demo for presentations
python week1/evaluate_xgboost.py    # confusion matrix + baseline verdict
```

## How to talk about metrics

An AUC/MAE here shows the model recovers relationships **you put into
synthetic data**, under a temporal split, with lift versus a simple
supplier baseline. That is a stronger interview answer than quoting a
single accuracy number.

## Tests

`tests/test_delay_model.py` checks prediction ranges (software
correctness), save/load round-trips, unseen categories, temporal split
usage, and baseline/diagnostic fields. `tests/test_split_dataset.py` and
`tests/test_evaluate_xgboost.py` cover the 60:20:20 export and ML-process
evaluation helpers.
