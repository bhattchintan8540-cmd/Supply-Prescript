# Requirement → Test Traceability

| Requirement | Primary tests | Notes |
|---|---|---|
| FR-1 Predict days + probability | `week1/tests/test_delay_model.py::test_predict_one_returns_sane_ranges`, `week3/tests/test_api.py::test_predict_returns_a_prediction` | Range checks = software correctness, not calibration proof |
| FR-2 Probability in expected cost | `week2/tests/test_solver.py::test_delay_launch_cost_scales_with_probability` | |
| FR-3 Fixed fees in MILP budget | `week2/tests/test_solver.py::test_fixed_fees_are_inside_budget_constraint` | |
| FR-4 Makespan vs weighted average | `test_makespan_mode_rejects_slow_channel_when_sla_is_tight`, `test_weighted_average_mode_meets_delay_ceiling` | |
| FR-5 Write-back + counterfactual | `week3/tests/test_api.py::test_full_decision_lifecycle_cost_accuracy_and_roi` | |
| FR-6 Outcome recording | same lifecycle test | |
| FR-7 Separate ROI vs cost accuracy | lifecycle test hits both endpoints | |
| FR-8 Outcomes → training | `week4/tests/test_retrain.py::test_outcomes_as_training_rows_uses_feature_snapshots` | |
| FR-9 Temporal + baselines | `week1/tests/test_delay_model.py::test_fit_uses_temporal_split_when_dates_present`, `test_fit_reports_baseline_and_classifier_diagnostics` | Val used for early stop / calibration / threshold |
| FR-10 Reality-based check parameters | `week1/tests/test_evaluate_xgboost.py` verdict tests | Bootstrap lift, recall/FPR/Brier/ECE gates |
| FR-11 Multi-signal drift | `week4/tests/test_retrain.py::test_should_retrain_triggers_on_hard_miss_even_if_cost_ok` | Cost MAPE alone is insufficient |

## Analytical validity vs software correctness

Automated tests prove the **software** implements the intended formulas
and contracts. They do **not** prove that predicted probabilities are
well-calibrated on real supply chains. Analytical validity is argued via
temporal validation, baselines, diagnostics, and honest synthetic-data
caveats in `/model/info` and these business docs.
