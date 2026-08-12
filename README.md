# SupplyPrescript

**Closed-loop prescriptive analytics for supply-chain delays**

Predict whether a shipment will be late → prescribe cost-aware options
(air freight, secondary supplier scenario, delay launch, or a PuLP MILP mix)
→ write the human decision back → log the real outcome → measure ROI vs
doing nothing → retrain when predictions drift, using eligible outcomes.

This is **Project 3** from the Axlero Solutions Data Analytics brief,
extended with deeper business/analytical logic (expected-cost decisions,
true ROI, operational delay constraints, outcome-aware retraining).

| Resource | Link |
|---|---|
| Problem statement | [PROJECT_BRIEF.md](PROJECT_BRIEF.md) |
| **Business analysis** | [docs/business/](docs/business/) |
| Beginner walkthrough | [STEP_BY_STEP.md](STEP_BY_STEP.md) |
| **VS Code / Cursor run guide** | [VSCODE.md](VSCODE.md) |
| Windows setup | [WINDOWS.md](WINDOWS.md) |
| **Presentation pack** (slides, demo script, Q&A) | [docs/presentation/](docs/presentation/) |
| EDA notebook | [notebooks/01_exploratory_analysis.ipynb](notebooks/01_exploratory_analysis.ipynb) |

---

## Architecture

```mermaid
flowchart LR
  A[Synthetic shipment history<br/>with dates + bad quarter] --> B[Feature builder]
  B --> C[XGBoost delay model<br/>prob + days / temporal val]
  C --> D[Prescriptive MILP<br/>expected cost + fixed fees]
  D --> E[FastAPI + dashboard]
  E --> F[(Decisions + features<br/>+ no-action cost)]
  F --> G[Outcome]
  G --> H[Cost accuracy + true ROI]
  H --> I{Drift > threshold?}
  I -->|yes| J[Retrain on shipments ∪ outcomes]
  J --> C
```

| Week | Folder | What you build |
|---|---|---|
| 1 | `week1/` | Synthetic data, EDA, features, XGBoost delay model (temporal validation), SQLAlchemy schema |
| 2 | `week2/` | Expected-cost formulas, PuLP MILP, HTML dashboard |
| 3 | `week3/` | FastAPI: prescribe → write-back → outcome → cost accuracy + ROI |
| 4 | `week4/` | Drift check + outcome-aware retrain |
| 5 | `week5/` | Confusion-matrix evaluation + closed-loop smoke |

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Week 1 — data, charts, model
python week1/generate_mock_data.py
python week1/explore_data.py
python week1/train_model.py

# Week 3 — API + dashboard (uses week2 UI + week1 model)
uvicorn week3.main:app --reload
```

Then open:

- Dashboard: http://127.0.0.1:8000/ui/
- Interactive API docs: http://127.0.0.1:8000/docs

Or use the Makefile: `make setup && make data && make explore && make train && make api`

**VS Code / Cursor:** open the repo root, select the `.venv` interpreter, then use
**Run and Debug** (API, train, demos, pytest) or **Tasks**. Guide: [VSCODE.md](VSCODE.md).

---

## Analytical design (read this before quoting metrics)

- **Probability is in the decision.** Expected holding = P(delay) × rate × days.
- **ROI needs a counterfactual.** `/decisions/roi` compares actual cost to Delay Launch (no action). Cost forecast error lives at `/decisions/cost-accuracy`.
- **Delay constraint matches operations.** Default = makespan (last unit). Weighted average only if partial fulfillment is useful.
- **Fixed fees are inside the MILP** via channel-activation binaries.
- **Metrics are on synthetic data** with programmed relationships, validated temporally against baselines. They show recovery of the synthetic environment — not real-world AUC claims.
- **Secondary supplier is scenario-based**, not a qualified supplier-selection engine.

Details: [docs/business/](docs/business/).

---

## Sample EDA output

After `python week1/explore_data.py`:

![Delay distribution](docs/figures/delay_distribution.png)

![Mean delay by supplier](docs/figures/delay_by_supplier.png)

![Peak-season effect](docs/figures/peak_season_effect.png)

Typical **synthetic / temporal-holdout** training metrics (seeded mock data):
MAE and AUC vary by split; always compare to the supplier-mean baseline in
`/model/info` and treat numbers as environment-recovery, not field accuracy.

---

## Presenting / live demo

```bash
# 1) Terminal — show the delay model on 4 scenarios
python week1/demo_model.py
python week2/demo_prescribe.py

# 2) Browser — click Demo A / B / C on the dashboard
uvicorn week3.main:app --reload
# → http://127.0.0.1:8000/ui/
```

Full click-by-click script: [docs/presentation/DEMO_SCRIPT.md](docs/presentation/DEMO_SCRIPT.md).
Slides: [docs/presentation/slides.html](docs/presentation/slides.html).

---

## Closed-loop API (Week 3)

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Delay days + late probability |
| `POST /prescribe` | Prediction + 4 options (expected costs use P(delay)) |
| `GET /model/info` | Temporal metrics, baselines, diagnostics (synthetic flag) |
| `POST /decisions` | **Write-back** — choice + feature snapshot + no-action cost |
| `PATCH /decisions/{id}/outcome` | **Close the loop** — log actual cost/delay |
| `GET /decisions/cost-accuracy` | Forecast error + budget adherence |
| `GET /decisions/roi` | **True ROI** vs Delay Launch counterfactual |
| `GET /health` | Liveness + whether the model is loaded |

---

## Drift-triggered retraining (Week 4)

```bash
python week4/retrain.py            # retrain only if cost drift is high
python week4/retrain.py --force    # always retrain
```

Retrain fits on `shipments.csv` **plus** resolved outcomes that have a
feature snapshot and actual delay label. It also refreshes `metrics.json`.
Reload the live API model afterward with `POST /model/reload`.

---

## Week 5 — evaluation + smoke

```bash
python week1/evaluate_xgboost.py   # confusion matrix + baseline verdict
python week5/smoke_loop.py         # data → train → prescribe → outcome
```

---

## Tests

```bash
python -m pytest -q
```

Tests prove **software correctness** (contracts, ranges, MILP constraints).
They do not by themselves prove analytical validity on real networks.

Shared fixture in `conftest.py` points every week’s tests at a throwaway SQLite DB.

---

## Skills demonstrated

- Translating a business decision into expected-cost optimization
- Predictive analytics (XGBoost classification + regression) with temporal validation
- Prescriptive analytics / MILP (PuLP) with fixed activation costs
- Operational write-back, true ROI, and cost-accuracy separation
- Outcome-aware drift monitoring and retrain trigger
- FastAPI + lightweight dashboard
- Business analysis docs with requirement→test traceability

---

## Optional Postgres

```bash
docker compose up -d
export DATABASE_URL=postgresql+psycopg2://sp_user:sp_pass@localhost:5432/supplyprescript
uvicorn week3.main:app --reload
```

Default storage is a local SQLite file (`data/supplyprescript.db`) — no Docker required for the demo path.
