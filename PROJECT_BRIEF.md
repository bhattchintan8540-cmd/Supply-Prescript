# Project 3 — SupplyPrescript (problem statement)

This is **Project 3** from the Axlero Solutions Data Analytics brief,
with analytical deepening documented in [docs/business/](docs/business/).

## What problem are we solving?

A supply-chain team often learns a shipment will be late *after* it is
already late. By then the options are expensive and messy.

**SupplyPrescript** closes that gap with a full analytics loop:

1. **Predict** — Will this shipment be late (probability), and by roughly how many days (magnitude)?
2. **Prescribe** — Given **expected financial impact** (P(delay) × cost of delay), what should we do?
   - Option A: **Air freight** (fast, expensive; fixed fee inside the MILP)
   - Option B: **Secondary supplier** (scenario-based backup option — not full supplier selection)
   - Option C: **Delay the launch** (no-action baseline; expected holding cost)
   - Option D: **Optimizer split** — a MILP mixes A/B/C to minimize expected cost under budget
     (including fixed fees) and an **operational delay** limit (makespan by default)
3. **Act + write back** — A human picks an option; we save that decision, feature snapshot, and no-action cost
4. **Close the loop** — Later we log what *actually* happened (cost & delay)
5. **Measure** — **Intervention ROI** vs Delay Launch; **cost accuracy** separately
6. **Learn** — If predictions drift, retrain on shipments **plus** eligible outcomes

## Why this matters (beginner view)

| Idea | Plain English |
|---|---|
| Predictive analytics | Guess the future from past data |
| Prescriptive analytics | Recommend the best action under constraints |
| Expected value | Use probability × impact, not impact alone |
| Closed loop | Record outcomes so the system can improve |
| Intervention ROI | Did acting beat doing nothing? |
| Cost accuracy | Did we forecast the chosen option's cost well? |

## Implementation phases (the brief's weeks)

| Week | Goal | Folder |
|---|---|---|
| 1 | Synthetic history + EDA + delay model + database scaffolding | `week1/` |
| 2 | Expected-cost formulas + PuLP MILP + dashboard UI | `week2/` |
| 3 | FastAPI: prescribe, write-back, outcome, cost accuracy + ROI | `week3/` |
| 4 | Drift check + outcome-aware retrain | `week4/` |

Follow **[STEP_BY_STEP.md](STEP_BY_STEP.md)** if you are new — it walks
through each commit and how to run the code yourself.

Business framing: **[docs/business/](docs/business/)**.
Presenting: **[docs/presentation/](docs/presentation/)**.
