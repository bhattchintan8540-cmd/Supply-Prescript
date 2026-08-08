# SupplyPrescript — one-pager (handout)

**Project 3 · Axlero Solutions Data Analytics**  
Closed-loop prescriptive analytics for supply-chain delays

---

## Problem
Teams often discover a shipment is late **after** it is already late. Panic expedites are costly; there is little structured comparison of options, and outcomes rarely feed back into the model.

## Solution loop
**Predict** delay risk → **Prescribe** four actions → **Write back** the human choice → **Log** actual cost/delay → **Retrain** when predictions drift.

| Option | Idea |
|---|---|
| A Air freight | Fast, expensive |
| B Secondary supplier | Medium cost / medium delay |
| C Delay launch | Cheap, accept delay |
| D Optimizer split | PuLP mix under budget + max delay |

## Build (4 weeks)
1. **Predict** — real USAID SCMS / UCI ingest → DB, EDA, XGBoost  
2. **Prescribe** — cost formulas, PuLP LP, dashboard  
3. **Act** — FastAPI write-back, outcomes, Decision ROI  
4. **Learn** — drift threshold + retrain trigger  

## Snapshot results (real open data)
- **~10,000** USAID SCMS shipments (or ~3.9k UCI Cargo 2000) seeded into SQLite/Postgres  
- Delay labels from scheduled vs actual delivery dates  
- Inspect `data/metrics.json` after training for MAE / AUC  

## Run the demo
```bash
pip install -r requirements.txt
python week1/ingest_real_data.py && python week1/train_model.py
uvicorn week3.main:app --reload
```
Dashboard: `http://127.0.0.1:8000/ui/`

## Skills shown
Real open-data ingest · relational store · EDA · feature engineering · predictive ML · prescriptive optimization · API / write-back · ROI & drift monitoring · reproducible GitHub project structure
