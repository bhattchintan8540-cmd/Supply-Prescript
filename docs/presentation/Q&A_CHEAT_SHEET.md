# Q&A cheat sheet

Short answers you can give if asked. Pair with [docs/business/](../business/).

### Why mock / synthetic data?
Real multi-year supplier lead-time logs are confidential. Synthetic data with supplier reliability, peak season, and a calendar bad quarter lets us demonstrate the full loop reproducibly. Metrics recover those programmed relationships — they are not field accuracy claims.

### Is MAE / AUC “good enough”?
Quote them only with the caveat above, and always versus the supplier-mean / late-rate **baselines** under a **temporal** split. For prescribing options (not auto-booking), lift vs baseline matters more than a raw AUC number.

### Why two models (classifier + regressor)?
Probability and magnitude answer different decisions. The optimizer uses both: expected holding = P(delay) × rate × days. High probability of a moderate delay ≠ low probability of a large one.

### Why isn’t probability just decorative?
It isn’t. Delay Launch and residual secondary holding scale with P(delay). Air residual delay is treated as nearly certain once that channel is activated.

### Why PuLP / MILP?
Prescriptive analytics needs constraints (budget including fixed fees, operational delay) and an objective (min expected cost). Binaries activate channels so fixed handling fees enter the model — not as a post-hoc patch.

### Why not always weighted-average delay?
If production waits for the last unit, operational delay is the **makespan**. Weighted average only makes sense when partial fulfillment creates usable value (`SP_PARTIAL_FULFILLMENT_USEFUL=1`).

### Does the system decide automatically?
No. It recommends; a human executes. That matches real ops governance.

### What is Decision / Intervention ROI?
Avoided loss = no-action (Delay Launch) cost − actual cost after intervention. ROI% = avoided loss / no-action cost. See `/decisions/roi`.

### What about the old “ROI” numbers (error %, on-budget %)?
Those are **cost forecast accuracy and budget adherence** (`/decisions/cost-accuracy`). Useful — but not ROI.

### What is drift?
Average relative cost error on resolved decisions. Above 15% → retrain trigger. Retrain fits on shipments **plus** outcomes that have feature snapshots.

### Is secondary supplier real selection?
No — it is a **scenario-based** intervention option for the prototype. Production would need qualification, capacity, MOQ, lead time, contracts.

### Software tests vs analytical validity?
Tests prove the code implements the contracts (ranges, constraints, API lifecycle). They do not prove probabilities are calibrated on real networks.

### Would this work with Postgres?
Yes — set `DATABASE_URL` and use `docker compose up -d`. SQLite is the zero-setup demo default.

### What’s missing for production?
Auth, richer inventory/MRP inputs, calibrated freight contracts, real alternate-supplier selection, monitoring/alerting, scheduler for retrain. See `docs/business/07_production_data_requirements.md`.

### How far did Phase 2 go?
Week 6 walks a 3×3 budget × delay grid. Week 7 names the cheapest feasible pure option at $100,000 and 5 days. Week 8 exports those files, smokes the path, and previews a decision with `POST /phase2/draft-decision`. That preview is in memory only — no decision row and no retrain.

### How far did Phase 3 go?
Phase 3 writes the chosen option back, logs the actual cost and delay, and compares that cost with Delay Launch. `GET /phase3/status` reports cost error, delay MAE, hard-miss rate, and outcome Brier. `POST /phase3/retrain` refits only when one of those signals is over its limit (15% cost error, 3 days, 35% hard-miss, Brier 0.25). Outcomes without a feature snapshot can raise drift but cannot become training rows. `python week4/finish_phase3.py` runs that loop on a throwaway database.

### Where is the code?
GitHub repo folders `week1`–`week8`, business docs in `docs/business/`, beginner guide `STEP_BY_STEP.md`.
