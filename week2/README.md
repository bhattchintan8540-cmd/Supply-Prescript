# Week 2 — Mathematical Optimization + Prescriptive UI

**Implementation Phase 2**: given a delay prediction, work out what to
actually do about it.

- `solver.py` — three cost formulas (air freight / secondary supplier /
  delay launch, matching the brief's Options A/B/C) plus a real PuLP
  linear program that blends all three channels to minimize cost under
  a budget cap and a max-delay constraint. See the module docstring for
  why the blended option is there and what it's trading off.
- `frontend/` — plain HTML/CSS/JS dashboard (no build step). Submits a
  shipment, shows the option cards, lets you "execute" one. Talks to
  the Week 3 API at `http://localhost:8000` by default.

## Run it

Open `frontend/index.html` directly in a browser once the Week 3 API
is running (`uvicorn week3.main:app --reload` from the project root).

## Tests

`tests/test_solver.py` is effectively the mid-review governance check:
the optimizer always fully allocates the order quantity, respects the
budget cap whenever a feasible solution exists (and clearly flags it
when it doesn't), and never returns a weighted average delay above the
requested ceiling.
