# SupplyPrescript

Closed-loop prescriptive analytics for supply chain delays: predict a
shipment's delay risk, get four mathematically-derived options for
handling it, act on one, and later find out whether the prediction was
actually right.

This is **Project 3** from the Axlero Solutions data analytics brief.
Start with [PROJECT_BRIEF.md](PROJECT_BRIEF.md) (problem statement) and
[STEP_BY_STEP.md](STEP_BY_STEP.md) (beginner walkthrough).

## Layout

```
week1/    predictive baseline + app scaffolding (data, model, db, config)
week2/    prescriptive solver (PuLP) + dashboard
week3/    FastAPI service - write-back and the closed-loop ROI endpoints
week4/    continuous learning / retrain trigger
data/     generated at runtime (csv, sqlite db, model artifact) - gitignored
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Week 1: generate mock shipment history + train the delay model
python week1/generate_mock_data.py
python week1/train_model.py

# Week 3: start the API (defaults to a local sqlite file, data/supplyprescript.db)
uvicorn week3.main:app --reload
```

Then open `week2/frontend/index.html` in a browser — it talks to
`http://localhost:8000`.

Run tests with `python -m pytest` from the project root.

### Switching to Postgres

```bash
docker compose up -d
export DATABASE_URL=postgresql+psycopg2://sp_user:sp_pass@localhost:5432/supplyprescript
uvicorn week3.main:app --reload
```
