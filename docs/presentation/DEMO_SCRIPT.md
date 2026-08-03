# Live demo script (3–5 minutes)

Do this **once at home** before the presentation so muscle memory is there.

## Before you present (setup)

```bash
cd Supply-Prescript          # your clone
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python week1/generate_mock_data.py
python week1/train_model.py
uvicorn week3.main:app --reload
```

Open two browser tabs:

1. **Dashboard:** http://127.0.0.1:8000/ui/
2. **API docs (backup):** http://127.0.0.1:8000/docs

Also open `docs/presentation/slides.html` for the deck.

---

## Script (say this while clicking)

### 1. Dashboard (10 sec)

> “This is the operations view. A planner enters a shipment and gets a prescription.”

### 2. Fill the form (keep defaults or use this)

| Field | Value |
|---|---|
| SKU | MICROCHIP-A2 |
| Supplier | NovaChip Manufacturing |
| Region | Asia Pacific |
| Distance | 8800 |
| Lead time | 16 |
| Quantity | 6000 |
| Unit cost | 14.2 |
| Budget | 95000 |
| Max delay | 5 |

Click **Get prescription**.

> “The model says roughly **X days** delay and about **Y%** chance it’s a meaningful delay.”

### 3. Option cards (30 sec)

Point to the four cards:

> “Air is fastest but costly. Secondary is a middle path. Delay launch is cheapest but accepts the slip.  
> The fourth card is the optimizer split — a math mix under our budget and delay limit.”

### 4. Execute decision (20 sec)

Click **Execute decision** on **Secondary Supplier** or **Optimizer Recommended Split**.

> “That write-back stores the human choice in the decisions table — the operational system of record.”

### 5. Log outcome (30 sec)

Click **Log outcome**. Enter something like:

- Actual cost: predicted cost × ~1.1 (e.g. if predicted was 94170, enter `103500`)
- Actual delay: `2`

> “Weeks later we learn the truth. Logging it closes the loop.”

### 6. ROI (20 sec)

Click **Refresh**.

> “Decision ROI shows how far predictions were from reality and how often we stayed on budget.  
> If that error stays high, Week 4’s retrain job fires.”

---

## Backup plan (if API won’t start)

1. Show EDA images in `docs/figures/`
2. In Swagger `/docs`, if API is up for you only — or show code snippets from `week3/main.py`
3. Show test command output: `python -m pytest -q` → 15 passed
4. Continue with Results / Skills slides

---

## Optional Week 4 one-liner

```bash
python week4/retrain.py
```

> “With few resolved decisions, it may say drift is under threshold — that’s correct behavior.”
