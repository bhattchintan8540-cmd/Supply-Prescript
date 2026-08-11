# Sample demo output (so you know what “good” looks like)

Captured from a trained model on the seeded **synthetic** data. Your
numbers will move after regenerating/retraining, but the **story** should
match: off-peak low risk → peak higher → risky supplier higher → worst
case highest. Absolute MAE/AUC are environment-recovery metrics, not
field claims.

## `python week1/demo_model.py`

Expect something in this shape (exact values vary):

```
Scenario                               Days  P(late>3d)  Risk bar
------------------------------------------------------------------------
Reliable supplier · off-peak           low        low
Same order · peak season               higher     higher
Risky supplier · long haul             higher     higher
Risky supplier · peak + long haul      highest    highest
```

## `python week2/demo_prescribe.py` (risky + peak)

```
expected delay : ~Xd
P(delay > 3d)  : ~Y%

Decision economics use P(delay)×magnitude for holding.
Delay constraint mode: operational makespan (default)

[ ? ] Air Freight                 $…   delay 1.0d   (fixed fee inside MILP)
[ ? ] Secondary Supplier          $…   delay ~0.35×days  (scenario option)
[ ? ] Delay Launch                $…   delay ~Xd     (no-action / ROI baseline)
[ ? ] Optimizer Recommended Split $…   operational delay ≤ SLA
```

Say out loud: probability changes Delay Launch expected cost; fixed fees
are inside the budget; makespan (not weighted average) is the default
operational constraint.

Use this if the live run fails — still explain the same narrative from this sheet.
See also `../business/10_design_decisions.md`.
