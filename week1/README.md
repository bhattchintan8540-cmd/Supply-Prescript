# Week 1 — Predictive Baseline + App Scaffolding

**Implementation Phase 1** of SupplyPrescript: everything needed before
there's anything to prescribe.

- `generate_mock_data.py` — builds a plausible 3-year shipment history
  (`../data/shipments.csv`) with a couple of deliberate "shock" patterns
  so the model has something real to learn instead of just noise.
- `explore_data.py` — beginner-friendly EDA: summary tables + charts in
  `docs/figures/`. Pair with `notebooks/01_exploratory_analysis.ipynb`.
- `features.py` — turns a shipment row into a numeric feature vector.
  Shared by both training and live inference so the two can't quietly
  drift apart from each other.
- `delay_model.py` — two small XGBoost models (a classifier for "will
  this be meaningfully late" and a regressor for "by how many days").
- `train_model.py` — CLI entry point: trains against the CSV, saves the
  artifact to `../data/delay_model.joblib` and metrics to
  `../data/metrics.json`.
- `demo_model.py` — **presentation demo**: compares four shipments so
  the audience can see supplier + peak-season effects live.
- `config.py` / `database.py` / `models.py` — the scaffolding every
  later week builds on: settings, the SQLAlchemy engine, and the ORM
  tables (`Shipment` for training data, `Decision` for the closed loop
  used from Week 3 onward).

## Run it

```bash
python week1/generate_mock_data.py
python week1/explore_data.py
python week1/train_model.py
python week1/demo_model.py          # live model demo for presentations
```

## Tests

`tests/test_delay_model.py` checks predictions land in sane ranges,
that saving/reloading the model round-trips identical predictions, and
that a supplier the model never saw during training doesn't crash
inference (onboarding a new vendor shouldn't break production).
