# Measurable Business Objective

## Primary objective

Minimize **expected fulfillment cost** while keeping **operational delay**
within a planner-defined SLA, subject to a budget that includes fixed
activation fees.

## Success measures (KPIs)

| KPI | Definition | Endpoint / source |
|---|---|---|
| Expected cost of recommended plan | MILP objective (variable + fixed fees), using P(delay)×holding for residual risk | `/prescribe` |
| Operational delay | Makespan (max delay across used channels) unless partial fulfillment is useful | solver result |
| Intervention ROI | `(no_action_cost − actual_cost) / no_action_cost` | `/decisions/roi` |
| Cost forecast accuracy | Mean absolute % error of predicted vs actual chosen-option cost | `/decisions/cost-accuracy` |
| Budget adherence | Share of resolved decisions with actual cost ≤ budget | `/decisions/cost-accuracy` |
| Model lift vs baseline | XGBoost MAE/AUC versus supplier-mean / late-rate baselines under temporal split | `/model/info` |

## Non-goals (prototype scope)

- Selecting a real secondary supplier (capacity, MOQ, qualification)
- End-to-end inventory / MRP integration
- Real-time carrier tracking
- Claiming real-world AUC/MAE from synthetic data
