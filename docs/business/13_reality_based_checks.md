# Reality-based check parameters and testing — DS / ML / Stats / AI

**Project:** SupplyPrescript (Axlero Project 3)  
**Question answered:** Do the parameters, tests, and metrics actually support
the idea — predict delay risk → prescribe under cost/SLA → measure ROI →
learn — or do soft pass criteria hide weak decision-grade quality?

## Executive verdict

The **product idea is sound**. The **science bar was soft**.

Before this hardening pass, a model could “pass” with near-baseline MAE,
a 50% false-alarm ceiling, TP > 0, and an unused validation split — while
still multiplying poorly calibrated `P(delay)` into money. That is not
ready for production-grade results; it is a demo that recovered synthetic
noise.

## What the project is trying to satisfy

| Layer | Intent | Must-have scientific object |
|---|---|---|
| Predictive | Late? + how many days? | Temporal generalization + calibrated probabilities |
| Prescriptive | Expected cost under budget/SLA | `P × impact` with honest residual-risk assumptions |
| Measurement | ROI vs no-action; cost accuracy separate | Counterfactual stored at decision time |
| Learning | Retrain on drift + outcomes | Multi-signal drift; anti-leakage dates |

## Problems found (reality-based review)

### 1. Validation set was unused
**Problem:** Docs advertised 60:20:20, but `fit()` never used the 20%
validation window for early stopping, calibration, or threshold tuning.  
**Why it hurts:** You cannot claim a validation strategy you do not use.  
**Fix:** Val now drives early stopping, isotonic calibration, and
FPR-constrained decision threshold. Test remains untouched.

### 2. Prior “model doing right” gates were too weak
**Problem:** Pass if AUC ≥ baseline, MAE ≤ baseline, TP > 0, FPR ≤ 50%.  
**Why it hurts:** Supplier-mean already explains most synthetic delay;
0.078d MAE lift and near-random segment AUCs (~0.54) still “passed.”  
**Fix:** Require AUC ≥ baseline+0.02, ≥3% relative MAE lift, recall ≥ 0.55,
FPR ≤ 35%, Brier ≤ 0.22, ECE ≤ 0.12, bootstrap AUC CI excluding 0.

### 3. No statistical significance on lift
**Problem:** Point estimates only — no CI / paired test.  
**Subjects connected:** inferential stats, bootstrap resampling, A/B style
model comparison.  
**Fix:** Bootstrap 95% CIs for AUC and MAE lift vs supplier baselines.

### 4. Probabilities entered finance without calibration
**Problem:** Brier ~0.21, fixed 0.5 threshold, no ECE.  
**Why it hurts:** Expected holding = `P × rate × days`. Mis-calibrated P
systematically mis-allocates air freight budget.  
**Subjects:** probability calibration, decision theory, proper scoring rules.  
**Fix:** Isotonic calibration on val; report Brier + ECE; persist threshold.

### 5. Incomplete metric matrix
**Missing before:** RMSE, R², PR-AUC, specificity, NPV, false-alarm rate,
ECE, bootstrap lifts, weak-segment map.  
**Why needed:** Classification vs regression tell different stories; PR-AUC
matters when late class is costly; segments expose Europe/supplier failure.

### 6. Hyperparameters were fixed without validation feedback
**Problem:** Fixed `n_estimators` / depth with no early stop or class weight.  
**Fix:** Early stopping on val; `scale_pos_weight`; light regularization
(`min_child_weight`, `reg_lambda`). Full grid search still optional for
portfolio scope — early stop is the minimum correct use of val.

### 7. Drift trigger was one-dimensional
**Problem:** Retrain only if cost MAPE ≥ 15%.  
**Why it hurts:** Cost can look fine while delay probabilities rot.  
**Fix:** Also trigger on hard-miss rate ≥ 40%, outcome Brier ≥ 0.30,
delay MAE ≥ 3 days.

### 8. Outcome retrain date leakage risk
**Problem:** `resolved_at` used as `shipment_date` → future timestamps in
temporal split.  
**Fix:** Prefer `created_at` (decision time) for temporal ordering.

### 9. Classifier ↔ regressor consistency unchecked
**Problem:** Can flag “late” while predicting 1.2 days (below 3-day rule).  
**Fix:** Consistency diagnostic on test; fail reality-based checks if violated.

### 10. Synthetic circularity still exists (honest limit)
**Problem:** Data programs supplier reliability, peak, bad quarter — models
recover that.  
**Rule:** Never claim real-world AUC/MAE. Keep `data_is_synthetic=true`.

## Reality-based check parameters and testing matrix

### Predictive (ML)

| Parameter / metric | Role | Gate / use |
|---|---|---|
| Temporal 60/20/20 | Out-of-time generalization | Required when dates exist |
| Early stopping (val) | Control overfit | Required |
| `scale_pos_weight` | Class imbalance | Required |
| Decision threshold (val, FPR-capped) | Operating point | Persist with artifact |
| Isotonic calibration | Trustworthy P for $ | Required for prescribe |
| AUC / PR-AUC | Ranking quality | AUC lift vs baseline + bootstrap |
| Precision / Recall / F1 / Specificity / NPV | Confusion economics | Recall ≥ 0.55; FPR ≤ 35% |
| Brier / ECE | Calibration | Brier ≤ 0.22; ECE ≤ 0.12 |
| MAE / RMSE / R² | Magnitude quality | ≥3% MAE lift vs supplier mean |
| Segment AUC | Fairness / ops risk | Warn if AUC < 0.60 |

### Statistical testing (connected subjects)

| Test / method | Subject link | Use here |
|---|---|---|
| Bootstrap CI on metric lift | Resampling / uncertainty | AUC & MAE vs baseline |
| Proper scoring (Brier) | Probabilistic forecast theory | Calibration health |
| ECE | Calibration diagnostics | Gate for money use of P |
| Confusion matrix / FPR / recall | Diagnostic testing (TPR/FPR) | Ops cost of FN vs FP |
| Temporal holdout | Forecasting hygiene | No random shuffle when dates exist |
| Baseline comparison | Null model / scientific control | Supplier-mean & late-rate |

### Prescriptive (OR / decision science)

| Parameter | Role | Gate |
|---|---|---|
| Expected holding `P×rate×days` | Risk-weighted no-action cost | Must use calibrated P |
| Makespan vs weighted delay | Operational semantics | Default makespan |
| Fixed fees in MILP | Budget integrity | Binaries in objective+budget |
| Budget relax flag | Honesty under infeasibility | Never hide overages |

### Closed-loop / MLOps

| Parameter | Role | Gate |
|---|---|---|
| Cost MAPE | Money forecast drift | ≥15% → retrain |
| Hard miss rate | Label/probability disagreement | ≥40% → retrain |
| Outcome Brier | Live calibration drift | ≥0.30 → retrain |
| Delay MAE | Magnitude drift | ≥3d → retrain |
| Feature snapshot at decision | Trainability of outcomes | Required to enter training frame |

## Ideas that *are* satisfied (keep)

1. Predict → prescribe → write-back → outcome → ROI vs Delay Launch → retrain.
2. Probability inside expected cost (not decoration).
3. ROI ≠ cost accuracy (separate endpoints).
4. Makespan default for “plant waits for last unit.”
5. Honest synthetic-data caveats in docs and `/model/info`.

## Remaining gaps (still not production-perfect)

| Gap | Why it still matters | Next move when real data arrives |
|---|---|---|
| No nested CV / rolling windows | One temporal cut can be lucky | Walk-forward validation |
| No cost-sensitive utility tuning | F1 ≠ $ loss of FN vs FP | Threshold by expected $ loss |
| No conformal prediction | No delay prediction intervals | Split-conformal intervals → SLA |
| Independent clf + reg | Soft consistency only | Multi-task / joint objective |
| Secondary supplier is scenario | Not true sourcing AI | Capacity, MOQ, qualification |
| Holding/$ fee constants illustrative | Wrong scales → wrong scripts | Calibrate from finance |

## Bottom line for reviewers

Confusion-matrix plots alone can look successful. Reality-based check
parameters — lift CIs, calibration, segment AUC, and real validation use —
show where quality was weak and what this pass hardens.

**Target result for this portfolio** is not a fake 0.95 AUC. It is:
scientifically honest metrics + decision-grade probabilities +
constraint-correct prescriptions + closed-loop measurement that finance
would recognize.
