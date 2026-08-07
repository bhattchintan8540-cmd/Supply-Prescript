# Project 3 — SupplyPrescript (problem statement)

This is **Project 3** from the Axlero Solutions Data Analytics brief.

## What problem are we solving?

A supply-chain team often learns a shipment will be late *after* it is
already late. By then the options are expensive and messy.

**SupplyPrescript** closes that gap with a full analytics loop:

1. **Predict** — Will this shipment be late, and by roughly how many days?
2. **Prescribe** — Given that prediction, what should we do?
   - Option A: **Air freight** (fast, expensive)
   - Option B: **Secondary supplier** (medium cost / medium delay)
   - Option C: **Delay the launch** (cheapest, accept the delay)
   - Option D: **Optimizer split** — a math model mixes A/B/C to minimize
     cost under a budget and a max-delay limit
3. **Act + write back** — A human picks an option; we save that decision
4. **Close the loop** — Later we log what *actually* happened (cost & delay)
5. **Learn** — If predictions drift too far from reality, retrain the model

## Why this matters (beginner view)

| Idea | Plain English |
|---|---|
| Predictive analytics | Guess the future from past data |
| Prescriptive analytics | Recommend the best action, not just a number |
| Closed loop | Record outcomes so the system can improve |
| Decision ROI | Did the AI's recommendations actually help? |

## Implementation phases (the brief's weeks)

| Week | Goal | Folder |
|---|---|---|
| 1 | Mock history + delay model + database scaffolding | `week1/` |
| 2 | Cost formulas + PuLP optimizer + dashboard UI | `week2/` |
| 3 | FastAPI: prescribe, write-back, outcome, ROI | `week3/` |
| 4 | Drift check + retrain trigger | `week4/` |

Follow **[STEP_BY_STEP.md](STEP_BY_STEP.md)** if you are new — it walks
through each commit and how to run the code yourself.
