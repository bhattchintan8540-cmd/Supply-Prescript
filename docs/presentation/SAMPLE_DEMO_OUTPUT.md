# Sample demo output (so you know what “good” looks like)

Captured from a model trained on USAID SCMS open data. Your numbers may
differ slightly after retraining, but the **story** should match:
dataset banner prints first → low-delay vendor → higher-risk vendor.

## `python week1/demo_model.py`

```
SupplyPrescript — delay model demo (real open data)
Dataset: USAID SCMS Delivery History
         10,307 rows · 73 suppliers · late>3d 9.7% · mean delay 2.46d

Scenario                                       Days  P(late>3d)  Risk bar
------------------------------------------------------------------------
Demo A · BRISTOL-MYERS SQUIBB                   0.x       low%
Demo B · Same supplier / peak                   0.x       …
Demo C · CIPLA LIMITED                         higher    higher
```

## `python week2/demo_prescribe.py` (Demo C / high-delay vendor)

```
Dataset: USAID SCMS … · 10,307 rows
expected delay : (from model)
P(delay > 3d)  : (from model)

[OK/OVER] Air Freight / Secondary / Delay Launch / Optimizer Split
```

## Dashboard (`http://127.0.0.1:8000/ui/`)

1. **Dataset panel** shows rows, suppliers, late rate, and `datasets/*.csv` files  
2. **EDA charts** from the real extract  
3. **Sample rows** with “Run model”  
4. **Demo A / B / C** buttons built from live `/dataset/demos`  
5. Prescribe → Execute → Log outcome → ROI  

Use this if the live run fails — still explain the same narrative from this sheet.
