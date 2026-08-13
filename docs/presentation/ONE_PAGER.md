# SupplyPrescript — one-pager (handout)

**Project 3 · Axlero Solutions Data Analytics**  
Closed-loop prescriptive analytics for supply-chain delays

---

## Problem
Teams often discover a shipment is late **after** it is already late. Panic expedites are costly; there is little structured comparison of options, and outcomes rarely feed back into the model.

## Solution loop
**Predict** delay risk → **Prescribe** expected-cost actions → **Write back** the human choice (+ no-action baseline) → **Log** actual cost/delay → **Measure** true ROI vs doing nothing → **Retrain** on shipments ∪ eligible outcomes when drift rises.

| Option | Idea |
|---|---|
| A Air freight | Fast, expensive; fixed fee inside MILP |
| B Secondary supplier | Scenario-based backup (not full supplier selection) |
| C Delay launch | No-action baseline; expected holding = P(delay)×impact |
| D Optimizer split | MILP mix under budget + operational delay (makespan default) |

## Build (4 weeks)
1. **Predict** — synthetic dated history, EDA, XGBoost + temporal validation  
2. **Prescribe** — expected-cost formulas, PuLP MILP, dashboard  
3. **Act** — FastAPI write-back, outcomes, cost accuracy + intervention ROI  
4. **Learn** — drift threshold + outcome-aware retrain  

## Snapshot results (seeded **synthetic** data)
- **~4,000** shipments with calendar bad-quarter shock  
- Metrics recover programmed relationships under temporal holdout — always compare to supplier baselines in `/model/info`  
- Pytest covers software correctness across the loop (not real-world calibration)

## Run the demo
```bash
pip install -r requirements.txt
python week1/generate_mock_data.py && python week1/train_model.py
uvicorn week3.main:app --reload
```
Dashboard: `http://127.0.0.1:8000/ui/`  
Business framing: `docs/business/`

## Skills shown
Business→math translation · EDA · feature engineering · predictive ML · MILP · API / write-back · true ROI · outcome-aware retraining · reproducible repo
