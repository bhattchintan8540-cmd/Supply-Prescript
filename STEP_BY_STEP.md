# Step-by-step guide (for beginners)

Each git commit on this branch matches one build step. Read this file
alongside `PROJECT_BRIEF.md`.

## Before you start (one-time setup)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You only need Python 3.10+. Postgres is optional (sqlite is the default).

Shortcut after setup: `make data`, `make explore`, `make train`, `make api`, `make test`.

**Prefer VS Code?** Follow the click-by-click beginner guide:
**[VSCODE.md](VSCODE.md)** (open folder → extensions → venv → train → F5 API).

---

## Step 0 — Scaffolding

Files: `README.md`, `PROJECT_BRIEF.md`, `requirements.txt`, `.gitignore`,
`docker-compose.yml`, `data/`.

Nothing runs yet — this just sets up the project shell.

---

## Step 1 — Week 1: predict delays

What you get:

- Fake 3-year shipment history (`generate_mock_data.py`)
- EDA charts (`explore_data.py` → `docs/figures/`) + notebook
- Feature builder shared by train + live predict (`features.py`)
- XGBoost classifier + regressor (`delay_model.py`)
- Database tables for shipments and decisions (`models.py`)

Run it:

```bash
python week1/generate_mock_data.py
python week1/explore_data.py
python week1/train_model.py
```

You should see:

1. ~4,000 rows written to `data/shipments.csv`
2. Four PNGs under `docs/figures/` (including Delta Cove bad quarter)
3. Training metrics printed (MAE ~1.8–2.0 days, AUC ~0.74 on seeded
   synthetic temporal holdout — always compare to the supplier baseline)
   and `data/delay_model.joblib` + `data/metrics.json` saved

Optional: open `notebooks/01_exploratory_analysis.ipynb` in VS Code / Jupyter.

---

## Step 2 — Week 2: prescribe actions

What you get:

- Three business options (air / secondary / delay launch)
- A real linear program (PuLP) that can split the order across channels
- A simple HTML dashboard in `week2/frontend/`

The dashboard is served by the Week 3 API at `/ui/` — open it after Step 3.

---

## Step 3 — Week 3: API + closed loop

Start the API (from the **project root**):

```bash
uvicorn week3.main:app --reload
```

Then open:

- [http://127.0.0.1:8000/ui/](http://127.0.0.1:8000/ui/) — dashboard
  (`http://127.0.0.1:8000` redirects here)
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) — interactive API docs

Try this loop once:

1. Submit a shipment → get 4 options
2. Click **Execute decision** on one option
3. Click **Log outcome** and enter a real cost/delay
4. Refresh measurement — Intervention ROI (vs no action) and cost accuracy appear

---

## Step 4 — Week 4: drift-triggered retrain

```bash
python week4/retrain.py            # only if drift is high
python week4/retrain.py --force    # always retrain
```

Retrain fits on shipments plus eligible outcomes that stored a feature snapshot.

---

## Step 5 — Week 5: evaluation + packaging

Confusion-matrix evaluation and end-to-end smoke check:

```bash
python week1/evaluate_xgboost.py
python week5/smoke_loop.py
python -m pytest -q
```

See `week5/README.md` for the evaluation deliverables folder.

---

## Step 6 — Tests

From the project root:

```bash
python -m pytest
```

All week folders share one `conftest.py` so tests use a throwaway sqlite DB.

---

## Presenting the project

Open the slide deck in a browser:

```text
docs/presentation/slides.html
```

Full pack (speaker notes, demo script, one-pager, Q&A):
[`docs/presentation/README.md`](docs/presentation/README.md).

---

## Common beginner mistakes

1. Running `uvicorn` from inside `week3/` — imports break. Stay at repo root.
2. Forgetting to generate data / train before calling `/predict`.
3. Opening the HTML file while the API is not running — the form will fail.
4. Expecting perfect forecasts — mock data + a small model will have error;
   the point is the full loop, not zero MAE.
