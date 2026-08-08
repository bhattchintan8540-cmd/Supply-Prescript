# Live demo script — show the model (5–7 minutes)

Practice this **once** before your presentation.

You have **two demo modes**:

| Mode | Best when | Command / URL |
|---|---|---|
| **A. Terminal model demo** | Projector / shared screen, no browser fuss | `python week1/demo_model.py` |
| **B. Browser dashboard** | You want the full prescribe + ROI story | http://127.0.0.1:8000/ui/ |

Most people should do **A then B** (2 min + 4 min).

---

## Setup (before the meeting)

```bash
cd Supply-Prescript
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python week1/ingest_real_data.py
python week1/train_model.py

# leave this running in a second terminal for Mode B
uvicorn week3.main:app --reload
```

Quick check:

```bash
python week1/demo_model.py
python week2/demo_prescribe.py
```

---

## Mode A — Terminal: “watch the model react” (≈2 min)

```bash
python week1/demo_model.py
```

**Say while the table prints:**

1. **Dataset banner** — “Loaded USAID SCMS — ~10k real shipments, not fake rows.”
2. **Demo A (low-delay vendor)** — “Calm baseline from the open extract.”
3. **Demo B · same supplier / peak** — “Only seasonality flipped.”
4. **Demo C (higher-delay vendor)** — “CIPLA-class corridor risk shows up as higher days / probability.”

Optional deepen:

```bash
python week2/demo_prescribe.py
```

> “Now we turn that forecast into four costed actions — including an optimizer split.”

---

## Mode B — Browser dashboard (≈4 min)

Open **http://127.0.0.1:8000/ui/**

### First — show the real dataset (30 sec)

Point at the top panel:

1. **Shipments / suppliers / late rate** — “This is USAID SCMS open data, not fake rows.”
2. **EDA charts** — delay distribution + supplier chart from the same extract.
3. **Sample rows** — optional: click **Run model** on one delayed row.

### Then — click the demo buttons (do not type)

1. Click **Demo A** (low-delay real vendor)  
   → Point at the big delay number and the green risk bar (should be lower).

2. Click **Demo B · Same supplier / peak**  
   → “Only seasonality changed.”

3. Click **Demo C** (higher-delay real vendor, e.g. CIPLA)  
   → “Highest risk in this trio. Now look at the four option cards.”

### Close the loop

4. On Demo C results, click **Execute decision** on **Optimizer Recommended Split** (or Secondary Supplier).
5. Click **Log outcome** — accept the suggested cost (~8% above predicted) and delay `2`.
6. Point at **Decision ROI** — error % and on-budget rate appear.

**Closing line:**

> “That’s the closed loop: the model predicted, we prescribed, a human chose, we logged reality, and we can retrain if we drift.”

---

## If something breaks live

| Problem | Fix |
|---|---|
| `Model not found` | `python week1/train_model.py` |
| Dashboard fetch fails | API not running — start `uvicorn week3.main:app --reload` from repo root |
| Port busy | `uvicorn week3.main:app --reload --port 8001` then open that port |
| No time for browser | Stay on Mode A only — still a strong model demo |

Backup slides: `docs/presentation/slides.html` (EDA charts still tell the story).
