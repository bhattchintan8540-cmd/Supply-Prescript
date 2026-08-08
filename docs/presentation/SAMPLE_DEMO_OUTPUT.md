# Sample demo output (so you know what “good” looks like)

Captured from a model trained on USAID SCMS open data. Your numbers may
differ slightly after retraining, but the **story** should match:
low-delay Europe vendor → higher-risk Asia→Africa ARV corridor.

## `python week1/demo_model.py`

```
Scenario                                     Days  P(late>3d)  Risk bar
------------------------------------------------------------------------
Trinity Biotech · Europe · off-peak           0.0       0.3%  [--------------------]
Same order · peak season                      0.0       0.3%  [--------------------]
Aurobindo · Asia→Africa · long haul          13.6      24.4%  [#####---------------]
CIPLA · Asia→Africa · peak                   17.2      20.2%  [####----------------]
```

## `python week2/demo_prescribe.py` (CIPLA / peak)

```
expected delay : ~17 days
P(delay > 3d)  : ~20%

[OK ] Air Freight                  ~$50,300   delay 1.0d
[OK ] Secondary Supplier           ~$3,090    delay ~6d
[OK ] Delay Launch                 ~$23,040   delay ~17d
[OK ] Optimizer Recommended Split  ~$13,342   delay 5.0d (mix of channels)
```

Use this if the live run fails — still explain the same narrative from this sheet.
