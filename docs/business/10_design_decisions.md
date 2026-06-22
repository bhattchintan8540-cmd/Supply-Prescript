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

## 6. Why retraining watches more than cost error

P(delay) scales expected holding in the MILP. Cost MAPE on the *chosen*
option can stay small while probabilities rot (wrong channel mix next
time). Retrain if **any** of these fire on resolved decisions:

| Signal | Default threshold | Question |
|---|---|---|
| Cost MAPE | 15% | Did we forecast the chosen option's cost? |
| Delay MAE | 3 days | Did magnitude predictions hold? |
| Hard-miss rate | 35% | Share of outcomes off by > 3 days |
| Outcome Brier | 0.25 | Did P(delay) match whether it was actually late? |

Training history comes from `shipments.csv` when present, otherwise the
seeded `Shipment` table, then eligible outcome snapshots.
