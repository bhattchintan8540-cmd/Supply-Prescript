# Sample demo output (so you know what “good” looks like)

Captured from a trained model on the seeded mock data. Your numbers may
differ slightly after retraining, but the **story** should match:
off-peak low risk → peak higher → risky supplier higher → worst case highest.

## `python week1/demo_model.py`

```
Scenario                               Days  P(late>3d)  Risk bar
------------------------------------------------------------------------
Reliable supplier · off-peak            1.2      18.2%  [####----------------]
Same order · peak season                4.3      72.2%  [##############------]
Risky supplier · long haul              5.8      73.4%  [###############-----]
Risky supplier · peak + long haul       8.3      95.9%  [###################-]
```

## `python week2/demo_prescribe.py` (risky + peak)

```
expected delay : ~8 days
P(delay > 3d)  : ~89%

[OVER] Air Freight                 ~$100,200   delay 1.0d
[OK ]  Secondary Supplier          ~$94,170    delay ~2.8d
[OK ]  Delay Launch                ~$88,100    delay ~8d
[OK ]  Optimizer Recommended Split ~$91,800    delay 5.0d (mix of channels)
```

Use this if the live run fails — still explain the same narrative from this sheet.
