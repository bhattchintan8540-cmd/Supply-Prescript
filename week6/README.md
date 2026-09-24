# Week 6 — Phase 2, to the midpoint

Week 2 already builds the four prescriptive options. Week 5 already
smokes the closed loop. There is no separate week-6 commit on `main`
before this folder.

This week runs **Phase 2 only halfway forward**: the same solver, on a
3×3 budget and max-delay grid around the demo shipment. It does not add
channels, write decisions, or retrain.

## Run it

```bash
python week6/phase2_midpoint.py
python week6/phase2_midpoint.py --budgets 80000,100000 --max-delays 5,8
python -m pytest week6 -q
```

You should see the grid rows, a write to `data/phase2_grid.csv`, and the
line `WEEK6 PHASE2 MIDPOINT OK`. Week 7 reads this same grid and names
the demo-point recommendation.
