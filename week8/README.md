# Week 8 — Phase 2 finish

Week 6 builds the budget × delay grid. Week 7 names the demo-point
option and can draft a decision in memory. This week finishes that
slice: export the files, smoke the path, and serve the recommendation
with optional `budget` and `max_delay` query parameters.

It still does not insert a decision row or retrain.

## Run it

```bash
python week7/recommend.py
python week8/smoke_phase2.py
python -m pytest week8 -q
```

You should see `data/phase2_grid.csv`, `data/phase2_recommendation.json`,
`draft_persisted=False`, and `WEEK8 PHASE2 FINISH OK`.

With the API running (`make api`):

```bash
curl http://127.0.0.1:8000/phase2/recommend
curl "http://127.0.0.1:8000/phase2/recommend?budget=120000&max_delay=8"
```
