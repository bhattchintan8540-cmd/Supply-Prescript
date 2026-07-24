# SupplyPrescript

Closed-loop prescriptive analytics for supply chain delays: predict a
shipment's delay risk, get four mathematically-derived options for
handling it, act on one, and later find out whether the prediction was
actually right.

This is Project 3 from the Axlero Solutions data analytics brief,
built separately from the MetricMind project - no shared code or
database between them.

## Layout

Each week's work lives in its own top-level folder, matching the
brief's week-wise implementation phases. Later weeks import from
earlier ones (Week 3's API needs Week 1's model and Week 2's solver) -
each week's `README.md` explains exactly what it pulls in and why.

```
week1/    predictive baseline + app scaffolding (data, model, db, config)
week2/    prescriptive solver (PuLP) + dashboard
week3/    FastAPI service - write-back and the closed-loop ROI endpoints
week4/    continuous learning / retrain trigger
data/     generated at runtime (csv, sqlite db, model artifact) - gitignored
```

Every week folder also has its own `tests/` - `conftest.py` sits at the
project root so pytest applies the same test-database setup to all of
them.

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

Run everything with `python -m pytest` from the project root (not
`cd` into a week folder first - the imports assume the root is on
`sys.path`, which running from there gives you for free).

Then open `week2/frontend/index.html` in a browser - it talks to
`http://localhost:8000`.

### Switching to Postgres

The app was designed against Postgres (see `docker-compose.yml`), but
defaults to sqlite so there's nothing to install before you can run
it. To use Postgres instead:

```bash
docker compose up -d
export DATABASE_URL=postgresql+psycopg2://sp_user:sp_pass@localhost:5432/supplyprescript
uvicorn week3.main:app --reload
```

Nothing else changes - `week1/database.py` reads `DATABASE_URL` and
the rest of the code doesn't know or care which engine is behind it.

## Why the solver returns four options, not three

The brief's example (air freight / secondary supplier / delay launch)
is three all-or-nothing strategies. `week2/solver.py` computes those
three as straightforward cost formulas, *and* runs an actual PuLP
linear program that splits the order across all three channels to
minimize cost subject to a budget cap and a max-delay constraint. That
blended answer usually beats every pure option and is the part that
justifies calling this "mathematical optimization" rather than three
if/else branches - see that file's docstring for the reasoning, and
`week2/tests/test_solver.py` for what's actually being enforced.

## Known simplifications

- `week4/retrain.py` re-fits on `data/shipments.csv` rather than
  folding resolved decisions directly into the training set - see
  `week4/README.md` for why.
- The dashboard in `week2/frontend/` is plain HTML/CSS/JS rather than
  a React build, since a build step doesn't add much at this size -
  the brief lists "Retool or React" as options anyway. Wiring a React
  front end up to the same API is a drop-in swap if you want one
  later.
- Fixed handling fees (the flat per-shipment cost on top of per-unit
  cost) aren't modeled *inside* the LP itself, since that needs binary
  "is this channel active" variables to do properly - overkill for
  three channels. They're added back onto the objective value after
  solving instead; see the comment in `solve_optimal_allocation`.
