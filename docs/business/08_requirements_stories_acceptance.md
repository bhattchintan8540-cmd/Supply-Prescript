# Requirements, User Stories, and Acceptance Criteria

## Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | System predicts delay days and P(significant delay) for a shipment |
| FR-2 | Expected holding cost uses P(delay) × rate × days for Delay Launch |
| FR-3 | Optimizer includes fixed activation fees via binary channel variables |
| FR-4 | Default delay constraint is operational makespan; weighted average only when partial fulfillment is useful |
| FR-5 | Human choice is written back with options snapshot, features, and no-action cost |
| FR-6 | Outcomes can be recorded later (actual cost, actual delay) |
| FR-7 | Cost accuracy and intervention ROI are reported as separate metrics |
| FR-8 | Retraining incorporates eligible resolved outcomes with feature snapshots |
| FR-9 | Model validation uses temporal split when dates exist, plus baselines and classifier diagnostics |

## User stories

1. **As a planner**, I want recommended options that reflect delay *probability*, so low-probability long delays are not treated like near-certain ones.
2. **As a plant manager**, I want the delay constraint to match whether partial receipts help production, so the math matches how the line actually starts.
3. **As finance**, I want ROI versus doing nothing, not just "prediction error called ROI."
4. **As an analyst**, I want outcomes to feed future training when features were captured, so retraining is not only re-fitting old data.
5. **As a reviewer**, I want synthetic-data metrics labeled as such, so AUC/MAE are not oversold.

## Acceptance criteria (selected)

| Story | Acceptance |
|---|---|
| Probability in decision | Identical magnitude, higher P(delay) ⇒ higher Delay Launch expected cost |
| Makespan | Under tight SLA, a slow channel cannot be mixed in when partial fill is false |
| Fixed fees | `within_budget` reflects variable + fixed fees from the MILP |
| True ROI | `/decisions/roi` uses `no_action_cost − actual_cost` |
| Closed-loop train | Resolved decisions with feature JSON appear in retrain training frame |
| Temporal validation | Metrics report `validation_strategy=temporal_70_15_15` when dates exist |
