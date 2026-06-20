# Week 4 — Drift Detection + Outcome-Aware Retraining

**Implementation Phase 4**: close the learning loop when enough
resolved decisions exist.

- `retrain.py` — pulls resolved decisions, computes average cost drift,
  and when drift crosses `RETRAIN_DRIFT_THRESHOLD` rebuilds the training
  frame from `data/shipments.csv` **plus** eligible outcomes that carry
  a shipment feature snapshot and an actual delay label.

```bash
python week4/retrain.py            # retrains only if drift is over threshold
python week4/retrain.py --force    # retrains unconditionally
```

## What "closed loop" means here

| Ingredient | Role |
|---|---|
| Cost drift from resolved decisions | Trigger |
| `shipment_features_json` on Decision | Features for new training rows |
| `actual_delay_days` on Decision | Label for new training rows |
| Temporal `DelayModel.fit` | Learns from history → predicts later periods |

Decisions logged **without** a feature snapshot still contribute to the
drift trigger, but cannot become training rows. The dashboard and API
now store features on execute so new decisions are eligible.

## Honest scope

This is a portfolio-scale closed loop: outcomes with snapshots are
folded into retraining. It is not an online learning system, and it
does not invent features for older decisions that lack snapshots.
