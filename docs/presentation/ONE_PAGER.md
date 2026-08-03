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
1. **Predict** — mock history, EDA, XGBoost classifier + regressor  
2. **Prescribe** — cost formulas, PuLP LP, dashboard  
3. **Act** — FastAPI write-back, outcomes, Decision ROI  
4. **Learn** — drift threshold + retrain trigger  

## Snapshot results (seeded mock data)
- **~4,000** shipments · **~46%** late &gt; 3 days  
- **MAE ≈ 1.87 days** · **AUC ≈ 0.79**  
- **15** pytest checks green  

## Run the demo
```bash
pip install -r requirements.txt
python week1/generate_mock_data.py && python week1/train_model.py
uvicorn week3.main:app --reload
```
Dashboard: `http://127.0.0.1:8000/ui/`

## Skills shown
EDA · feature engineering · predictive ML · prescriptive optimization · API / write-back · ROI & drift monitoring · reproducible GitHub project structure
