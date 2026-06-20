# Assumptions, Constraints, and Business Rules

## Assumptions

1. Historical shipment attributes in `data/shipments.csv` are **synthetic**,
   with programmed supplier reliability, peak seasonality, and a calendar
   bad quarter for Delta Cove.
2. Holding cost is linear at `$0.06` per unit per delay day (illustrative).
3. Air freight residual delay ≈ 1 day once activated (nearly certain).
4. "Secondary supplier" is a **scenario-based intervention option**, not a
   real supplier-selection engine (no capacity, MOQ, or qualification).
5. By default, production cannot start until the **last** unit arrives
   (`partial_fulfillment_useful=False` → makespan constraint).
6. A human remains accountable for the final choice.

## Constraints

| Constraint | Representation |
|---|---|
| Fulfill full order quantity | `Σ x_k = Q` |
| Budget including fixed fees | `Σ (c_k x_k + f_k y_k) ≤ B` |
| Operational delay SLA | makespan `≤ D_max` (default) or weighted avg |
| Optional on-time fill | `Σ x_k for on-time channels ≥ α Q` |
| Channel activation | `x_k ≤ Q y_k`, `y_k ∈ {0,1}` |

## Business rules

- Delay Launch is the **no-action counterfactual** stored for ROI.
- Classifier probability enters expected holding for Delay Launch and
  residual secondary risk; air residual delay is treated as certain.
- If the budget-constrained MILP is infeasible, solve without budget and
  flag `budget_relaxed` / `within_budget=false` — do not hide overages.
- Retrain when average cost drift ≥ 15% (configurable).
