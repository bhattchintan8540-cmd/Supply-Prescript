# Google Slides outline (copy/paste)

Use with speaker notes in `SPEAKER_NOTES.md` and business framing in `../business/`.

**Slide 1 — Title**  
- SupplyPrescript  
- Closed-loop delay decisions: predict → prescribe → decide → measure → retrain  

**Slide 2 — Problem**  
- Late discovery → expensive panic expedites  
- No structured options, no counterfactual ROI  

**Slide 3 — Decision loop**  
- Prediction → Recommendation → Human Decision → Outcome → Measurement → Retraining  

**Slide 4 — Why probability matters**  
- Expected holding = P(delay) × rate × days  
- Classifier is part of the decision, not decoration  

**Slide 5 — Options**  
- A Air freight — fast; fixed fee inside MILP  
- B Secondary supplier — scenario-based backup  
- C Delay launch — no-action baseline  
- D Optimizer split — expected-cost MILP under budget & operational delay  

**Slide 6 — Architecture**  
- Week 1: dated synthetic data, EDA, XGBoost + temporal validation  
- Week 2: expected costs, PuLP MILP, UI  
- Week 3: FastAPI, decisions, cost accuracy + true ROI  
- Week 4: drift + outcome-aware retrain  

**Slide 7 — Data**  
- 4,000 synthetic shipments with shipment dates  
- Calendar bad quarter for Delta Cove  
- Peak-season effect  

**Slide 8 — EDA (insert images)**  
- `docs/figures/delay_distribution.png`  
- `docs/figures/delay_by_supplier.png`  

**Slide 9 — Peak insight**  
- Insert `docs/figures/peak_season_effect.png`  

**Slide 10 — Model**  
- Shared feature builder  
- Temporal 60/20/20 validation  
- Compare MAE/AUC to supplier baselines  
- Metrics recover synthetic relationships — not field claims  

**Slide 11 — Prescribe**  
- Expected-cost formulas using P(delay)  
- MILP: fill order, budget incl. fixed fees, makespan (or weighted avg if partial fill helps)  

**Slide 12 — Closed loop API**  
- POST /prescribe  
- POST /decisions (write-back + no-action cost + features)  
- PATCH outcome  
- GET /decisions/roi (true ROI)  
- GET /decisions/cost-accuracy  

**Slide 13 — Retraining**  
- Drift = average cost error  
- Retrain if drift ≥ 15% on shipments ∪ eligible outcomes  

**Slide 14 — Demo**  
- /ui/ → prescribe → execute → log outcome → ROI + cost accuracy  

**Slide 15 — Results**  
- Synthetic temporal metrics + baseline lift  
- Software tests green; analytical validity argued separately  

**Slide 16 — Skills**  
- Business→math translation, ML, MILP, API, true ROI, closed-loop retrain  

**Slide 17 — Thank you / Q&A**  
- Repo + `docs/business/` + demo URL  
