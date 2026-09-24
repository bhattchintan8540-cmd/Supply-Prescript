# Week 7 — Phase 2 midpoint recommendation

Week 6 sweeps the existing prescriptive solver. This week reads that
grid and stops at the demo operating point: budget $100,000 and a 5-day
delay ceiling.

It names the cheapest pure option that is inside both limits, and how
many neighboring cells agree. It does not save a decision or retrain.

## Run it

```bash
python week7/recommend.py
python -m pytest week7 -q
```

You should see the chosen option, a grid table, a write to
`data/phase2_recommendation.json`, and `WEEK7 PHASE2 MIDPOINT OK`.
