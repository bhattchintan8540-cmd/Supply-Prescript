# Speaker notes — SupplyPrescript presentation

Use with [`slides.html`](slides.html). Target time: **8–12 minutes** + demo + Q&A.

Open the deck: double-click `slides.html` or open it from VS Code / browser.  
**Keys:** `→` next · `←` prev · `F` fullscreen · `P` print to PDF.

---

## Slide 1 — Title (20 sec)

> “This is SupplyPrescript — Project 3 from the Axlero data analytics brief.  
> It is a **closed-loop** system: we don’t just predict delay, we recommend what to do, record the decision, and learn from the outcome.”

## Slide 2 — Agenda (20 sec)

Walk the six bullets quickly. Say you’ll do a short live demo near the end.

## Slide 3 — Problem (60–90 sec)

Tell a story:

> “Imagine a product launch depends on a chip shipment.  
> Today, many teams find out it’s late when the truck doesn’t show up.  
> Then everyone scrambles to air-freight — which is expensive — or they slip the launch.  
> The business problem is **timing**: if we knew earlier, we could choose the right trade-off.”

## Slide 4 — Solution loop (60 sec)

Point to each box: Predict → Prescribe → Act → Outcome → Retrain.

> “Predictive tells us *what might happen*. Prescriptive tells us *what to do about it*.  
> Closing the loop means we compare prediction to reality and improve.”

## Slide 5 — Four options (45 sec)

Name A/B/C/D. Emphasize:

> “A human still decides. The system is a decision-support tool, not autopilot.”

## Slide 6 — Architecture (45 sec)

Map weeks to layers. Mention tech stack only at the bottom line.

## Slide 7 — Real datasets (60 sec)

> “Two public sources. USAID SCMS is the default ~10k health-commodity shipments.  
> UCI Cargo 2000 is the ~3.9k freight set that replaces the old fake 4k rows.  
> Both live under `datasets/` and seed the shipments table.”

## Slide 8 — Training extract (45 sec)

> “On SCMS we see ~10k rows, 73 suppliers, about 10% late beyond 3 days, with a long tail.”

## Slides 9–10 — EDA (60–90 sec)

Show charts:

1. Distribution — “most shipments are on time; the long tail is what hurts.”
2. By supplier — “CIPLA and Aurobindo sit higher on the Asia→Africa corridor; Trinity / BMS are near-zero.”
3. Peak season — “real seasonality can differ from intuition — always check the data.”

## Slide 11 — Model (60 sec)

> “Two XGBoost models share one feature builder so training and live inference can’t drift apart.  
> Classifier: chance of a meaningful delay. Regressor: how many days.  
> On held-out SCMS data we typically see AUC around 0.80 — good enough to drive options, not perfect.”

## Slide 11 — Prescribe (45 sec)

> “Week 2 turns the forecast into money and days.  
> Three pure options plus a small linear program that can split the order across channels under budget and max-delay constraints.”

## Slide 12 — API / closed loop (45 sec)

> “Week 3 is the operational layer: prescribe, save the choice, later patch the real outcome, then read Decision ROI.”

## Slide 13 — Retrain (30 sec)

> “If average cost error drifts past 15%, we retrain.  
> That’s continuous learning in portfolio form.”

## Slide 14 — Demo (live)

Follow [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md). If network/API fails, skip to Results and show figures + `/docs` screenshots.

## Slide 15 — Results (30 sec)

Hit the three metrics and “15 tests passing.”

## Slide 16 — Skills (30 sec)

Pick 3–4 that match the interview/course rubric (EDA, ML, optimization, closed loop).

## Slide 17 — Q&A

Invite questions. Have [`Q&A_CHEAT_SHEET.md`](Q&A_CHEAT_SHEET.md) open on a second screen.

---

## Timing cheat sheet

| Block | Minutes |
|---|---|
| Problem + solution | 3 |
| Weeks 1–4 story | 4 |
| Live demo | 3–5 |
| Results + close | 1 |
| Q&A | remaining |
