# Week 4 — Continuous Learning

**Implementation Phase 4**: the piece that actually makes this
"closed-loop" instead of just "logs outcomes and forgets them."

- `retrain.py` — pulls resolved decisions, computes the average drift
  between predicted and actual cost, and retrains the Week 1 model if
  that drift crosses `RETRAIN_DRIFT_THRESHOLD` (see `week1/config.py`).
  Not a scheduler itself — point cron, a GitHub Action, or a Retool
  scheduled query at it.

```bash
python week4/retrain.py            # retrains only if drift is over threshold
python week4/retrain.py --force    # retrains unconditionally
```

## Known simplification

A `Decision` row doesn't carry the full original shipment feature set,
only enough to compute cost drift — so the retrain step re-fits on
`data/shipments.csv` rather than folding resolved decisions directly
into the training data. A production version would join `Decision`
back to `Shipment` via a foreign key and actually grow the training set
over time. Documented here rather than hidden, since it's the one part
of the closed loop that's more "demonstrates the trigger mechanism"
than "actually learns from every decision."
