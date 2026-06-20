# Design Decisions (analyst notes)

How each mathematical construct maps to the business problem.
For the fuller BA pack see the sibling files in this folder.

## 1. Why the classifier feeds the optimizer

| Quantity | Role |
|---|---|
| P(significant delay) | Scales expected holding if we do nothing |
| Predicted delay days | Magnitude of the delay being risked |
| Expected holding | `P × $0.06/unit/day × days` |

Without probability, a 20% chance of a 7-day slip looks identical to a
95% chance of a 7-day slip. That is not how a planner should spend
expedite budget.

## 2. Why ROI ≠ cost prediction error

| Metric | Formula | Question |
|---|---|---|
| Cost accuracy | `|actual − predicted| / predicted` | Did we forecast the chosen option well? |
| Intervention ROI | `(no_action − actual) / no_action` | Did intervening beat doing nothing? |

## 3. Why makespan beats weighted average by default

If 5,000 units arrive in 1 day and 5,000 in 9 days, the weighted average
is 5 — but a plant that needs all 10,000 cannot start until day 9.
Makespan is the operational delay unless partial fulfillment is useful.

## 4. Why fixed fees are decision variables

Post-hoc fees can make a "within budget" LP solution over budget after
the fact. Channel binaries put `fee_k * y_k` in the objective and budget.

## 5. Why retraining uses outcome snapshots

Drift on resolved decisions is the trigger. Learning requires labels
*and* features. Feature JSON on Decision lets eligible outcomes join the
training frame; older rows without snapshots only affect the trigger.
