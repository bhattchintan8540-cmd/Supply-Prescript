# Data Dictionary

## Training shipments (`shipments` / `data/shipments.csv`)

| Field | Type | Description |
|---|---|---|
| shipment_date | ISO date | Shipment date for temporal validation |
| sku | string | Component SKU |
| supplier | string | Origin supplier name |
| origin_region | string | Region (drives distance baseline) |
| distance_km | float | Approximate lane distance |
| historical_avg_lead_time_days | float | Historical lead-time feature |
| order_quantity | int | Units ordered |
| unit_cost_usd | float | Unit cost |
| is_peak_season | bool | Nov/Dec peak flag |
| actual_delay_days | float | Label — days late vs plan |

## Decisions (operational write-back)

| Field | Type | Description |
|---|---|---|
| shipment_sku | string | SKU at decision time |
| predicted_delay_days | float | Regressor output |
| predicted_delay_probability | float | P(delay > 3 days) |
| options_json | text | Snapshot of offered options |
| shipment_features_json | text | Feature snapshot for retraining |
| chosen_option_label | string | Human choice |
| predicted_cost_usd | float | Expected cost of chosen option |
| no_action_cost_usd | float | Delay Launch counterfactual |
| budget_cap_usd | float | Budget at decision time |
| actual_cost_usd | float | Observed cost (nullable until resolved) |
| actual_delay_days | float | Observed delay (nullable until resolved) |

## Intervention channel parameters (illustrative)

See constants in `week2/solver.py` (air surcharge/fee, secondary premium/fee,
holding cost per unit-day).
