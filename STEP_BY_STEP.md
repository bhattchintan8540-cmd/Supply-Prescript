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

---

## Step 0 — Scaffolding

Files: `README.md`, `PROJECT_BRIEF.md`, `requirements.txt`, `.gitignore`,
`docker-compose.yml`, `data/`.

Nothing runs yet — this just sets up the project shell.

---

## Step 1 — Week 1: predict delays

What you get:

- Fake 3-year shipment history (`generate_mock_data.py`)
- Feature builder shared by train + live predict (`features.py`)
- XGBoost classifier + regressor (`delay_model.py`)
- Database tables for shipments and decisions (`models.py`)

Run it:

```bash
python week1/generate_mock_data.py
python week1/train_model.py
```

You should see a MAE (average day error) and AUC printed, and a file at
`data/delay_model.joblib`.

---

## Step 2 — Week 2: prescribe actions

What you get:

- Three business options (air / secondary / delay launch)
- A real linear program (PuLP) that can split the order across channels
- A simple HTML dashboard in `week2/frontend/`

The dashboard needs the Week 3 API, so open it after Step 3.

---

## Step 3 — Week 3: API + closed loop

Start the API (from the **project root**):

```bash
uvicorn week3.main:app --reload
```

Then either:

- open `http://127.0.0.1:8000/docs` (interactive API docs), or
- open `week2/frontend/index.html` in a browser

Try this loop once:

1. Submit a shipment → get 4 options
2. Click **Execute decision** on one option
3. Click **Log outcome** and enter a real cost/delay
4. Refresh ROI — you should see average error % appear

---

## Step 4 — Week 4: continuous learning

```bash
python week4/retrain.py            # only if drift is high
python week4/retrain.py --force    # always retrain
```

This is the "if predictions are wrong often, retrain" switch.

---

## Step 5 — Tests

From the project root:

```bash
python -m pytest
```

All week folders share one `conftest.py` so tests use a throwaway sqlite DB.

---

## Common beginner mistakes

1. Running `uvicorn` from inside `week3/` — imports break. Stay at repo root.
2. Forgetting to generate data / train before calling `/predict`.
3. Opening the HTML file while the API is not running — the form will fail.
4. Expecting perfect forecasts — mock data + a small model will have error;
   the point is the full loop, not zero MAE.
