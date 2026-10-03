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

## Step 6 — Week 6: Phase 2 midpoint sweep

```bash
python week6/phase2_midpoint.py
python week6/phase2_midpoint.py --budgets 80000,100000 --max-delays 5,8
```

Grid rows, `data/phase2_grid.csv`, then `WEEK6 PHASE2 MIDPOINT OK`.

---

## Step 7 — Week 7: Phase 2 midpoint recommendation

```bash
python week7/recommend.py
```

The cheapest feasible pure option at the demo budget and delay ceiling,
a grid table, `data/phase2_recommendation.json`, then
`WEEK7 PHASE2 MIDPOINT OK`.

---

## Step 8 — Week 8: Phase 2 finish

```bash
python week8/finish_phase2.py
```

That command does not need GNU `make`. It exports the grid and the
recommendation, checks the in-memory draft, and calls the Phase 2 API
routes. It prints `WEEK8 PHASE2 COMPLETE OK`.

`make phase2-smoke` is the same smoke step when `make` is installed.

With the API up (`make api`):

```bash
curl http://127.0.0.1:8000/phase2/recommend
curl "http://127.0.0.1:8000/phase2/recommend?budget=120000&max_delay=8"
```

Smoke prints `WEEK8 PHASE2 FINISH OK` and `draft_persisted=False`. The
GET route returns the same demo-point winner without training.

Phase 2 completion checklist:

- [ ] `python week6/phase2_midpoint.py` writes `data/phase2_grid.csv`
- [ ] `python week7/recommend.py` writes `data/phase2_recommendation.json`
- [ ] `python week8/finish_phase2.py` prints `WEEK8 PHASE2 COMPLETE OK`
- [ ] `GET /phase2/grid` returns nine cells
- [ ] `GET /phase2/recommend` names the demo-point winner
- [ ] `POST /phase2/draft-decision` returns `persisted: false` and adds no decision row
- [ ] Dashboard shows the Phase 2 budget × delay grid

---

## Step 9 — Phase 3: close the loop

```bash
python week4/finish_phase3.py
```

That uses a throwaway database. It prescribes Air Freight, writes the
decision, logs an outcome, then reads ROI and drift. It prints
`WEEK4 PHASE3 COMPLETE OK`. It does not force a model refit.

With the API up:

```bash
curl http://127.0.0.1:8000/phase3/status
curl -X POST http://127.0.0.1:8000/phase3/retrain -H "Content-Type: application/json" -d "{\"force\": false}"
```

The dashboard section **Phase 3 — close the loop** shows the same
signals. **Refresh** reloads them. **Retrain if drift is high** refits
only when a signal is over its limit.

Phase 3 completion checklist:

- [ ] `python week4/finish_phase3.py` prints `WEEK4 PHASE3 COMPLETE OK`
- [ ] `GET /phase3/status` returns the four drift thresholds
- [ ] `POST /phase3/retrain` with `{"force": false}` returns `retrained`
- [ ] A resolved outcome without `shipment_features_json` can raise drift and does not become a training row
- [ ] The dashboard shows resolved decisions, the drift line, Refresh, and Retrain if drift is high

---

## Step 10 — Tests

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
