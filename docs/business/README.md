# Business Analysis Index

This folder is the analytical bridge between the business problem and
the Python implementation. Read these before polishing slides further.

| Doc | Contents |
|---|---|
| [01_business_problem.md](01_business_problem.md) | Decision process and problem statement |
| [02_objectives_and_kpis.md](02_objectives_and_kpis.md) | Measurable objectives and KPI definitions |
| [03_stakeholders.md](03_stakeholders.md) | Who cares and why |
| [04_current_vs_future_state.md](04_current_vs_future_state.md) | Manual process → closed loop |
| [05_assumptions_constraints_rules.md](05_assumptions_constraints_rules.md) | Assumptions, MILP constraints, rules |
| [06_data_dictionary.md](06_data_dictionary.md) | Field definitions |
| [07_production_data_requirements.md](07_production_data_requirements.md) | What a production version would need |
| [08_requirements_stories_acceptance.md](08_requirements_stories_acceptance.md) | FR, stories, acceptance criteria |
| [09_requirement_test_traceability.md](09_requirement_test_traceability.md) | Requirement → test map |
| [10_design_decisions.md](10_design_decisions.md) | Why each math construct matches the business |
| [11_dataset_analysis.md](11_dataset_analysis.md) | Train / data validation / testing 60:20:20 split |
| [12_xgboost_ml_process.md](12_xgboost_ml_process.md) | XGBoost ML process, verdict, confusion matrix (TP/FP) |

## Interview prompts these docs answer

1. Why is delay probability used by the optimizer?
2. Why is ROI not "cost prediction error"?
3. Why isn't weighted-average delay always the right constraint?
4. Why are fixed fees inside the MILP?
5. What does retraining learn from newly captured outcomes?
