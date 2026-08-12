# Dataset analysis

**Data Analytics by Axlero — SupplyPrescript**

Copy this folder (or the generated CSVs) to your machine at:

`C:\Users\ACER\Desktop\Data Analytics by Axlero`

## Split ratio

| Split | Role | Share |
|---|---|---|
| **Train** | Fit the delay model | **60%** |
| **Data validation** | Tune / check before final scoring | **20%** |
| **Testing** | Hold-out evaluation | **20%** |

## How to generate (local files only)

```bash
python week1/generate_mock_data.py
python week1/split_dataset.py
```

Or open `notebooks/02_dataset_analysis.ipynb`.

## Outputs (not committed to GitHub)

These live under `data/` and are listed in `.gitignore`:

- `data/train.csv` — Train (60%)
- `data/validation.csv` — Data validation (20%)
- `data/test.csv` — Testing (20%)
- `data/dataset_split_summary.json` — counts and date windows

The delay model (`week1/delay_model.py`) uses the same **temporal 60:20:20** strategy when `shipment_date` is present.

**How each slice is used (non-negotiable):**

| Split | Used for |
|---|---|
| Train (60%) | Fit XGBoost trees only |
| Validation (20%) | Early stopping, isotonic probability calibration, decision-threshold tuning |
| Test (20%) | Final metrics / reality-based check parameters only — never tuning |
