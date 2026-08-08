# Q&A cheat sheet

Short answers you can give if asked.

### Why real open data (not mock)?
Default ingest pulls USAID SCMS delivery history (and optionally UCI Cargo 2000). Those are public shipment logs with scheduled vs actual delivery — enough to train a delay model and seed a real `shipments` table. `make data-mock` remains for fully offline demos.

### Is MAE ~1.9 days “good enough”?
For prescribing *options* (not auto-booking), yes as a baseline. The point of the closed loop is to measure and improve with real outcomes.

### Why two models (classifier + regressor)?
Probability and magnitude answer different decisions. High probability of a small delay ≠ certain large delay.

### Why PuLP / linear programming?
Prescriptive analytics needs constraints (budget, max delay) and an objective (min cost). An LP is the clear, explainable way to blend channels.

### Does the system decide automatically?
No. It recommends; a human executes. That matches real ops governance.

### What is Decision ROI?
After outcomes are logged: average predicted vs actual cost, average error %, and % of decisions that stayed within budget.

### What is drift?
Average relative cost error on resolved decisions. Above 15% → retrain trigger.

### Would this work with Postgres?
Yes — set `DATABASE_URL` and use `docker compose up -d`. SQLite is the zero-setup demo default.

### What’s missing for production?
Auth, richer shipment↔decision linkage for training on outcomes, calibrated freight costs, monitoring/alerting, and a scheduler for retrain.

### Where is the code?
GitHub repo `Supply-Prescript`, folders `week1`–`week4`, beginner guide `STEP_BY_STEP.md`.
