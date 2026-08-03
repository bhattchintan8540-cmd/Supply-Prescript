# Google Slides / PowerPoint outline

Copy each block into its own slide. Keep titles short; paste body as bullets.

---

**Slide 1 — Title**  
SupplyPrescript  
Closed-loop analytics for supply-chain delays  
Axlero · Data Analytics · Project 3

**Slide 2 — Agenda**  
- Problem  
- Solution loop  
- Week-by-week build  
- Live demo  
- Results & skills  
- Q&A  

**Slide 3 — Problem**  
- We learn shipments are late *after* they are late  
- Panic air freight is expensive  
- No clear speed vs cost vs delay trade-offs  
- Outcomes rarely improve the next forecast  

**Slide 4 — Solution**  
Predict → Prescribe → Act/write-back → Log outcome → Retrain on drift  

**Slide 5 — Four options**  
- A Air freight — fast, costly  
- B Secondary supplier — medium / medium  
- C Delay launch — cheap, accept slip  
- D Optimizer split — PuLP mix under budget & max delay  

**Slide 6 — Architecture**  
- Week 1: data, EDA, XGBoost  
- Week 2: costs, PuLP, UI  
- Week 3: FastAPI, decisions, ROI  
- Week 4: drift + retrain  

**Slide 7 — Data**  
- 4,000 mock shipments  
- 5 suppliers · peak-season effect  
- ~46% late beyond 3 days  

**Slide 8 — EDA (insert images)**  
- `docs/figures/delay_distribution.png`  
- `docs/figures/delay_by_supplier.png`  

**Slide 9 — Peak insight**  
- Insert `docs/figures/peak_season_effect.png`  
- Off-peak ~2.7d vs peak ~5.5d  

**Slide 10 — Model**  
- Shared feature builder  
- Classifier AUC ≈ 0.79  
- Regressor MAE ≈ 1.87 days  

**Slide 11 — Prescribe**  
- Business cost formulas  
- PuLP LP: fill order, budget, max delay, min cost  

**Slide 12 — Closed loop API**  
- POST /prescribe  
- POST /decisions (write-back)  
- PATCH outcome  
- GET /decisions/roi  

**Slide 13 — Continuous learning**  
- Drift = average cost error  
- Retrain if drift ≥ 15%  

**Slide 14 — Demo**  
- /ui/ → prescribe → execute → log outcome → ROI  

**Slide 15 — Results**  
- MAE ~1.87d · AUC ~0.79 · 15 tests passing  

**Slide 16 — Skills**  
- EDA, ML, optimization, API, ROI/monitoring, reproducible repo  

**Slide 17 — Thank you / Q&A**  
- Repo + demo URL  
