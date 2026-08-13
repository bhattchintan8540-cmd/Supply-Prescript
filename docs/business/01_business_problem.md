# Business Problem

## Context

A manufacturing planner must decide how to fulfill a component order when
a shipment is at risk of arriving late. Doing nothing can idle a
production line. Expediting everything by air can blow the budget.
Splitting to a backup channel may help — but only if the operational
constraint (when production can actually start) is modeled correctly.

## Decision process this project supports

```text
Prediction → Recommendation → Human Decision → Actual Outcome → Measurement → Retraining
```

That is the business process. A delay probability alone is not a decision.
A cost-minimizing allocation that ignores when production can start is
not a decision either. The system exists to support a human choice under
budget and service constraints, then learn from what happened.

## Problem statement

> Given a pending shipment's attributes, estimate the risk and magnitude
> of delay, recommend cost-aware fulfillment options (including a
> constrained optimal mix), capture the human choice and later outcome,
> measure whether intervening beat doing nothing, and retrain when
> outcomes drift from predictions.

## Why this is not "train a model and stop"

Portfolio projects often end at `model.predict(X)`. The business does not.
Somebody has to choose air freight, a backup option, or delay the launch —
and finance will ask whether that choice created value versus the
no-action baseline.
