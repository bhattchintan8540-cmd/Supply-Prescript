# Presentation pack

Everything you need to present **SupplyPrescript** (Project 3).

| File | Use it for |
|---|---|
| [`slides.html`](slides.html) | Main slide deck (open in Chrome/Edge, press `F` for fullscreen) |
| [`SPEAKER_NOTES.md`](SPEAKER_NOTES.md) | What to say on each slide + timing |
| [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | Click-by-click live demo (terminal + browser) |
| [`ONE_PAGER.md`](ONE_PAGER.md) | Print/PDF handout for evaluators |
| [`Q&A_CHEAT_SHEET.md`](Q&A_CHEAT_SHEET.md) | Short answers to likely questions |
| [`GOOGLE_SLIDES.md`](GOOGLE_SLIDES.md) | Copy-paste outline into Google Slides / PowerPoint |
| [`../business/`](../business/) | Business problem → math → KPI traceability |

Before presenting, skim `../business/10_design_decisions.md` so you can
defend probability-in-the-objective, true ROI, makespan, fixed fees, and
outcome-aware retraining.

## Show the model in 30 seconds

```bash
python week1/generate_mock_data.py   # once
python week1/train_model.py          # once
python week1/demo_model.py           # ← run this on stage
```

Then open the dashboard and click **Demo A / B / C**:

```bash
uvicorn week3.main:app --reload
# http://127.0.0.1:8000/ui/
```

## Quick start on presentation day

1. Open `slides.html` in a browser → press **F** (fullscreen).
2. Keep `SPEAKER_NOTES.md` on a second screen or printed.
3. Start the API and follow `DEMO_SCRIPT.md` for the live part.
4. Print `ONE_PAGER.md` (or File → Print from any Markdown preview) as a leave-behind.

## Export slides to PDF

With `slides.html` open: press **P** (print) → Destination **Save as PDF** → enable **Background graphics**.
