# Week 1 — Predictive Baseline + App Scaffolding

**Implementation Phase 1** of SupplyPrescript: everything needed before
there's anything to prescribe.

- `ingest_real_data.py` — downloads real open shipment history (USAID
  SCMS by default, or UCI Cargo 2000), maps it onto the model schema,
  writes `../data/shipments.csv`, and **seeds the `shipments` table**
  in SQLite/Postgres.
- `data_adapters.py` — source-specific download + transform logic.
- `generate_mock_data.py` — offline fallback only (`make data-mock`).
- `explore_data.py` — beginner-friendly EDA: summary tables + charts in
  `docs/figures/`. Pair with `notebooks/01_exploratory_analysis.ipynb`.
- `features.py` — turns a shipment row into a numeric feature vector.
  Shared by both training and live inference so the two can't quietly
  drift apart from each other.
- `delay_model.py` — two small XGBoost models (a classifier for "will
  this be meaningfully late" and a regressor for "by how many days").
- `train_model.py` — CLI entry point: trains against the DB (preferred)
  or CSV, saves the artifact to `../data/delay_model.joblib` and
  metrics to `../data/metrics.json`.
- `demo_model.py` — **presentation demo**: compares four shipments so
  the audience can see supplier + peak-season effects live.
- `config.py` / `database.py` / `models.py` — the scaffolding every
  later week builds on: settings, the SQLAlchemy engine, and the ORM
  tables (`Shipment` for training data, `Decision` for the closed loop
  used from Week 3 onward).

## Run it

```bash
python week1/ingest_real_data.py      # real USAID SCMS → CSV + DB
# python week1/ingest_real_data.py --source uci-c2k   # ~3.9k UCI freight rows
# python week1/ingest_real_data.py --source both
python week1/explore_data.py
python week1/train_model.py
python week1/demo_model.py            # live model demo for presentations
```

Offline without network: `python week1/generate_mock_data.py`

## Real data sources

| Source | Flag | Rows (approx) | What it is |
|---|---|---|---|
| USAID SCMS Delivery History | `--source usaid-scms` (default) | ~10k | Public PEPFAR health-commodity shipments with vendors, modes, scheduled vs actual delivery |
| UCI Cargo 2000 | `--source uci-c2k` | ~3.9k | Real air-freight process instances from the UCI ML Repository |

## Tests

`tests/test_delay_model.py` checks predictions land in sane ranges,
that saving/reloading the model round-trips identical predictions, and
that a supplier the model never saw during training doesn't crash
inference. `tests/test_real_data.py` covers the SCMS/C2K adapters and
DB seed round-trip without needing the network.
