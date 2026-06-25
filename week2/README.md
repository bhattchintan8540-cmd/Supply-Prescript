# Week 2 — Mathematical Optimization + Prescriptive UI

**Implementation Phase 2**: given a delay prediction *and* a delay
probability, work out what to actually do about it.

- `solver.py` — three expected-cost formulas (air freight / secondary
  supplier / delay launch) plus a PuLP **MILP** that blends channels to
  minimize expected cost under a budget (including fixed activation fees)
  and an **operational delay** constraint.
- `frontend/` — plain HTML/CSS/JS dashboard (no build step). Includes
  **Demo A / B / C** one-click scenarios for live presentations.
  Submits a shipment, shows the option cards, lets you "execute" one.
  Served by the Week 3 API at `http://127.0.0.1:8000/ui/`.
- `demo_prescribe.py` — terminal version of predict + prescribe for
  when you cannot share a browser.

## Decision logic (interview-ready)

| Design choice | Why |
|---|---|
| Expected holding = P(delay) × rate × days | Classifier is part of the decision, not decoration |
| Fixed fees inside the MILP via binaries | Budget must reflect real modeled cost |
| Makespan delay by default | Production often waits for the last unit; weighted average only if partial fulfillment is useful |
| Secondary supplier is scenario-based | Prototype assumes a backup option exists; not a real supplier-selection engine |

Toggle weighted-average mode with `SP_PARTIAL_FULFILLMENT_USEFUL=1` when
arriving units create usable value before the full order lands.

## Run it

With the Week 3 API running (`uvicorn week3.main:app --reload` from the
project root), open http://127.0.0.1:8000/ui/ — or open `frontend/index.html`
directly in a browser (it will call `http://localhost:8000`).

## Tests

`tests/test_solver.py` checks: full order fulfillment, budget including
fixed fees, probability-scaled Delay Launch cost, makespan vs weighted
average delay modes, and activation of only SLA-feasible channels.
