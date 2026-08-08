# SupplyPrescript

**Closed-loop prescriptive analytics for supply-chain delays**

Predict whether a shipment will be late → prescribe cost-aware options
(air freight, secondary supplier, delay launch, or a PuLP-optimized mix)
→ write the human decision back → log the real outcome → retrain when
predictions drift.

This is **Project 3** from the Axlero Solutions Data Analytics brief,
built week-by-week so a beginner can follow every commit.

| Resource | Link |
|---|---|
| Problem statement | [PROJECT_BRIEF.md](PROJECT_BRIEF.md) |
| Beginner walkthrough | [STEP_BY_STEP.md](STEP_BY_STEP.md) |
| Windows setup | [WINDOWS.md](WINDOWS.md) |
| **Presentation pack** (slides, demo script, Q&A) | [docs/presentation/](docs/presentation/) |
| EDA notebook | [notebooks/01_exploratory_analysis.ipynb](notebooks/01_exploratory_analysis.ipynb) |

---

## Architecture

```mermaid
flowchart LR
  A[Real open shipment history<br/>USAID SCMS / UCI C2K] --> B[Feature builder]
  B --> C[XGBoost delay model<br/>prob + days]
  C --> D[Prescriptive solver<br/>PuLP LP + 3 pure options]
  D --> E[FastAPI + dashboard<br/>week3 / week2]
  E --> F[(Decisions table<br/>write-back)]
  F --> G[Outcome + ROI]
  G --> H{Drift > threshold?}
  H -->|yes| C
```

| Week | Folder | What you build |
|---|---|---|
| 1 | `week1/` | Real open data ingest → DB, EDA, features, XGBoost delay model |
| 2 | `week2/` | Cost formulas, PuLP optimizer, HTML dashboard |
| 3 | `week3/` | FastAPI: prescribe → write-back → outcome → ROI |
| 4 | `week4/` | Drift check + retrain trigger |

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Week 1 — real data, charts, model
python week1/ingest_real_data.py   # USAID SCMS (~10k) → CSV + shipments table
python week1/explore_data.py
python week1/train_model.py

# Week 3 — API + dashboard (uses week2 UI + week1 model)
uvicorn week3.main:app --reload
```

Then open:

- Dashboard: http://127.0.0.1:8000/ui/
- Interactive API docs: http://127.0.0.1:8000/docs

Or use the Makefile: `make setup && make data && make explore && make train && make api`

Offline fallback (no network): `make data-mock` instead of `make data`.

### Real data sources

| Source | Command | Approx rows |
|---|---|---|
| **USAID SCMS** delivery history (default) | `python week1/ingest_real_data.py` | ~10,000 |
| **UCI Cargo 2000** freight tracking | `python week1/ingest_real_data.py --source uci-c2k` | ~3,900 |
| Both concatenated | `python week1/ingest_real_data.py --source both` | ~14,000 |

Training history is stored in the SQLAlchemy `shipments` table (SQLite by
default, Postgres via `DATABASE_URL`) and mirrored to `data/shipments.csv`.

---

## Sample EDA output

After `python week1/explore_data.py`:

![Delay distribution](docs/figures/delay_distribution.png)

![Mean delay by supplier](docs/figures/delay_by_supplier.png)

![Peak-season effect](docs/figures/peak_season_effect.png)

Training metrics depend on the source extract; inspect `data/metrics.json`
after `train_model.py`.

---

## Presenting / live demo

```bash
# 1) Terminal — show the delay model on 4 scenarios
python week1/demo_model.py
python week2/demo_prescribe.py

# 2) Browser — click Demo A / B / C on the dashboard
uvicorn week3.main:app --reload
# → http://127.0.0.1:8000/ui/
```

Full click-by-click script: [docs/presentation/DEMO_SCRIPT.md](docs/presentation/DEMO_SCRIPT.md).
Slides: [docs/presentation/slides.html](docs/presentation/slides.html).

---

## Closed-loop API (Week 3)

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Delay days + late probability |
| `POST /prescribe` | Prediction + 4 options (A/B/C + optimizer split) |
| `GET /model/info` | Training metrics (MAE, AUC, top features) for demos |
| `POST /decisions` | **Write-back** — persist the chosen option |
| `PATCH /decisions/{id}/outcome` | **Close the loop** — log actual cost/delay |
| `GET /decisions/roi` | Decision ROI (predicted vs actual) |
| `GET /health` | Liveness + whether the model is loaded |

---

## Continuous learning (Week 4)

```bash
python week4/retrain.py            # retrain only if cost drift is high
python week4/retrain.py --force    # always retrain
```

---

## Tests

```bash
python -m pytest -q
```

Shared fixture in `conftest.py` points every week’s tests at a throwaway SQLite DB.

---

## Skills demonstrated

- Real open-data ingest (USAID SCMS / UCI) into a relational store
- Predictive analytics (XGBoost classification + regression)
- Feature engineering shared by train and live inference
- Prescriptive analytics / linear programming (PuLP)
- Operational write-back and Decision ROI
- Drift monitoring and retrain trigger
- FastAPI + lightweight dashboard
- Pytest coverage across all four weeks

---

## Optional Postgres

```bash
docker compose up -d
export DATABASE_URL=postgresql+psycopg2://sp_user:sp_pass@localhost:5432/supplyprescript
python week1/ingest_real_data.py   # seeds the shipments table in Postgres
uvicorn week3.main:app --reload
```

Default storage is a local SQLite file (`data/supplyprescript.db`) — no Docker required for the demo path.
