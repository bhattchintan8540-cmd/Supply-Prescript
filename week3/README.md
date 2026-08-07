# Week 3 — Write-Back, Closed Loop, Decision ROI

**Implementation Phase 3**: wire the model and solver up to a real API,
and start closing the loop the brief describes — persist which option
someone picked, and later log what actually happened.

- `schemas.py` — request/response models for the API.
- `main.py` — the FastAPI app. Imports the model + database layer from
  `week1` and the solver from `week2` (see the sys.path note at the top
  of the file for how that cross-folder import is wired up).

  | Endpoint | What it does |
  |---|---|
  | `POST /predict` | raw delay prediction for a shipment |
  | `GET /model/info` | training metrics (MAE / AUC / top features) |
  | `POST /prescribe` | prediction + the four options from Week 2 |
  | `POST /decisions` | **write-back** — persists the option someone chose |
  | `GET /decisions` | list everything logged so far |
  | `PATCH /decisions/{id}/outcome` | **closes the loop** — record what actually happened |
  | `GET /decisions/roi` | Decision ROI: predicted vs. actual cost, on-budget rate |

## Run it

```bash
uvicorn week3.main:app --reload
```

(run from the project root, not from inside `week3/`, so `week1.`/`week2.`
imports resolve)

Then open http://127.0.0.1:8000/ui/ for the dashboard or
http://127.0.0.1:8000/docs for interactive API docs.

## Tests

`tests/test_api.py` walks the full lifecycle: prescribe → pick an
option → write it back → log an outcome → check it shows up in the ROI
summary. Also checks that picking an option that wasn't actually
offered gets rejected (422), not silently accepted.
