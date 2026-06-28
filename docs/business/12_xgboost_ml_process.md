# Dataset analysis — ML Process for XGBoost

**Data Analytics by Axlero — SupplyPrescript**

## What this covers

1. XGBoost ML process (features → 60:20:20 temporal split → regularized fit with early stopping → evaluate)
2. Verdict: **is the model doing right?** (lift vs supplier baselines **and** balanced train/val capacity)
3. Confusion matrix plot
4. True Positive / False Positive annotated matrix (TN / FP / FN / TP)

Training uses shallow trees, L2 / min-child / subsample regularization, class
`scale_pos_weight`, and a validation split: earlier val for early stopping
(plus at most one over/under-fit capacity bump), later val for probability
calibration and the decision threshold. Test metrics stay untouched.

## Generate copy-ready files (not committed to GitHub)

```bash
python week1/generate_mock_data.py
python week1/evaluate_xgboost.py
```

Then copy (or unzip) to your PC:

`C:\Users\ACER\Desktop\Data Analytics by Axlero`

| Local path | Purpose |
|---|---|
| `exports/Data Analytics by Axlero/` | Ready folder to copy as-is |
| `exports/Data_Analytics_by_Axlero_XGBoost.zip` | Zip — extract onto Desktop |
| `data/ml_evaluation/` | Same plots/JSON (working copy) |

All of the above are **gitignored** so they do not appear as GitHub modifications.

## Notebook

`notebooks/03_xgboost_ml_process.ipynb`
