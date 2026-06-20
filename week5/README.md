# Week 5 — Evaluation, packaging, and demo smoke

**Implementation Phase 5** of SupplyPrescript: prove the ML process,
package evaluation artifacts, and smoke-test the closed loop.

## What you get

- Confusion-matrix evaluation via `week1/evaluate_xgboost.py`
  (TN / FP / FN / TP plots + baseline verdict)
- 60:20:20 dataset analysis docs + notebooks
- `smoke_loop.py` — data → train → prescribe → decision → outcome → drift
- Tests for the smoke path

## Run it

```bash
python week1/generate_mock_data.py
python week1/train_model.py
python week1/evaluate_xgboost.py
python week5/smoke_loop.py
python -m pytest week5 -q
```

Evaluation outputs land under `data/ml_evaluation/` and (optionally)
`exports/` — both are gitignored. Copy the export folder locally for
Axlero deliverables.

## How to talk about Week 5

Week 5 answers: *Is the model doing the right thing vs a naive baseline,
and can we walk the full loop without the demo falling over?* Metrics
remain synthetic-environment recovery, not field claims.
