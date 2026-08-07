"""
Week 1 - the predictive half of SupplyPrescript.

Two small XGBoost models sharing one feature set:
  - a classifier answering "will this ship late enough to matter?"
    (delay > features.DELAY_FLAG_THRESHOLD_DAYS)
  - a regressor answering "by roughly how many days?"

Both feed the Week 2 optimizer: expected holding cost uses
P(delay) × magnitude, so the classifier is part of the decision —
not a decorative metric.

Validation & scientific controls
--------------------------------
When `shipment_date` is present, fit() uses a temporal split
(60% train / 20% validation / 20% test):
  - train: fit trees
  - validation: early stopping, probability calibration, decision threshold
  - test: final metrics only (never used for tuning)

Statistical honesty: AUC/MAE lifts are reported with bootstrap 95% CIs.
A "pass" requires the model to beat baselines *and* show lift whose CI
does not cross zero (when sample size allows). Metrics recover synthetic
programmed relationships — not real-world AUC claims.
"""
from __future__ import annotations

from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from .features import DELAY_FLAG_THRESHOLD_DAYS, build_features


def _temporal_split_masks(dates: pd.Series, train_frac: float = 0.60, val_frac: float = 0.20):
    """Return boolean masks for train / val / test ordered by time (60:20:20)."""
    order = dates.argsort(kind="mergesort")
    n = len(dates)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _safe_auc(y_true, y_prob) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _safe_pr_auc(y_true, y_prob) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def _expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float | None:
    """ECE — how far predicted probabilities sit from observed frequencies."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0 or pd.Series(y_true).nunique() < 2:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / total) * abs(acc - conf)
    return float(ece)


def _bootstrap_lift_ci(
    y_true,
    model_scores,
    baseline_scores,
    metric: str,
    n_boot: int = 400,
    seed: int = 13,
) -> dict:
    """Bootstrap 95% CI for (model − baseline) lift.

    For AUC: higher is better → lift = model_auc − baseline_auc.
    For MAE: lower is better → lift = baseline_mae − model_mae.
    """
    y_true = np.asarray(y_true)
    model_scores = np.asarray(model_scores, dtype=float)
    baseline_scores = np.asarray(baseline_scores, dtype=float)
    n = len(y_true)
    if n < 30 or pd.Series(y_true).nunique() < 2:
        return {"lift": None, "ci_low": None, "ci_high": None, "significant": None, "n_boot": 0}

    rng = np.random.default_rng(seed)
    lifts = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if metric == "auc":
            if pd.Series(yt).nunique() < 2:
                continue
            m = roc_auc_score(yt, model_scores[idx])
            b = roc_auc_score(yt, baseline_scores[idx])
            lifts.append(float(m - b))
        elif metric == "mae":
            m = mean_absolute_error(yt, model_scores[idx])
            b = mean_absolute_error(yt, baseline_scores[idx])
            lifts.append(float(b - m))
        else:
            raise ValueError(metric)

    if len(lifts) < 50:
        return {"lift": None, "ci_low": None, "ci_high": None, "significant": None, "n_boot": len(lifts)}

    arr = np.asarray(lifts)
    point = float(np.mean(arr))
    lo, hi = np.percentile(arr, [2.5, 97.5])
    # Significant if 95% CI does not cross zero.
    significant = bool(lo > 0)
    return {
        "lift": round(point, 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "significant": significant,
        "n_boot": int(len(lifts)),
    }


def _tune_threshold(y_true, y_prob, max_fpr: float = 0.35) -> float:
    """Pick decision threshold on validation under an FPR ceiling.

    Supply-chain FN (missed late) is costly, but unchecked FPR burns
    expedite budget. Maximize F1 among thresholds with FPR ≤ max_fpr;
    if none qualify, pick the lowest FPR that still finds some positives.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if pd.Series(y_true).nunique() < 2:
        return 0.5

    candidates = []
    for t in np.linspace(0.20, 0.80, 61):
        pred = (y_prob >= t).astype(int)
        tn = int(((y_true == 0) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        score = f1_score(y_true, pred, zero_division=0)
        candidates.append((float(t), float(score), float(fpr), tp))

    feasible = [c for c in candidates if c[2] <= max_fpr and c[3] > 0]
    if feasible:
        best = max(feasible, key=lambda c: c[1])
        return best[0]

    # No threshold meets the FPR ceiling — choose minimal FPR with TP>0.
    with_tp = [c for c in candidates if c[3] > 0]
    if not with_tp:
        return 0.5
    best = min(with_tp, key=lambda c: (c[2], -c[1]))
    return best[0]


def _group_metric(y_true, y_prob, y_pred, groups, metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, idx in groups.items():
        if len(idx) == 0:
            continue
        yt = y_true.iloc[idx] if hasattr(y_true, "iloc") else y_true[idx]
        if metric == "auc":
            val = _safe_auc(yt, y_prob[idx])
        elif metric == "precision":
            val = float(precision_score(yt, y_pred[idx], zero_division=0))
        elif metric == "recall":
            val = float(recall_score(yt, y_pred[idx], zero_division=0))
        else:
            continue
        if val is not None:
            out[str(name)] = round(val, 3)
    return out


class DelayModel:
    def __init__(self):
        self.classifier: XGBClassifier | None = None
        self.regressor: XGBRegressor | None = None
        self.categories: dict[str, list[str]] | None = None
        self.feature_columns: list[str] | None = None
        # Supplier-mean delay learned on the training window (baseline).
        self._supplier_mean_delay: dict[str, float] | None = None
        self._global_mean_delay: float = 0.0
        self._supplier_late_rate: dict[str, float] | None = None
        self._global_late_rate: float = 0.0
        # Validation-tuned decision threshold + isotonic calibrator for P(delay).
        self.decision_threshold: float = 0.5
        self._calibrator: IsotonicRegression | None = None

    def _calibrate_proba(self, raw_proba: np.ndarray) -> np.ndarray:
        raw_proba = np.asarray(raw_proba, dtype=float)
        if self._calibrator is None:
            return np.clip(raw_proba, 0.0, 1.0)
        return np.clip(self._calibrator.predict(raw_proba), 0.0, 1.0)

    def _raw_proba(self, X) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Classifier not trained")
        return self.classifier.predict_proba(X)[:, 1]

    # -- training -----------------------------------------------------
    def fit(self, shipments: pd.DataFrame, verbose: bool = True) -> dict:
        shipments = shipments.reset_index(drop=True)
        X, self.categories = build_features(shipments)
        self.feature_columns = X.columns.tolist()
        y_days = shipments["actual_delay_days"].reset_index(drop=True)
        y_flag = (y_days > DELAY_FLAG_THRESHOLD_DAYS).astype(int)

        has_dates = "shipment_date" in shipments.columns
        if has_dates:
            dates = pd.to_datetime(shipments["shipment_date"])
            train_mask, val_mask, test_mask = _temporal_split_masks(dates)
            X_train, X_val, X_test = X.loc[train_mask], X.loc[val_mask], X.loc[test_mask]
            y_days_train, y_days_val, y_days_test = (
                y_days.loc[train_mask],
                y_days.loc[val_mask],
                y_days.loc[test_mask],
            )
            y_flag_train, y_flag_val, y_flag_test = (
                y_flag.loc[train_mask],
                y_flag.loc[val_mask],
                y_flag.loc[test_mask],
            )
            meta_train = shipments.loc[train_mask]
            meta_test = shipments.loc[test_mask]
            validation_strategy = "temporal_60_20_20"
            n_val = int(val_mask.sum())
        else:
            # Random 60:20:20 fallback when dates are missing.
            (
                X_temp,
                X_test,
                y_days_temp,
                y_days_test,
                y_flag_temp,
                y_flag_test,
                meta_temp,
                meta_test,
            ) = train_test_split(
                X, y_days, y_flag, shipments, test_size=0.20, random_state=13
            )
            (
                X_train,
                X_val,
                y_days_train,
                y_days_val,
                y_flag_train,
                y_flag_val,
                meta_train,
                _meta_val,
            ) = train_test_split(
                X_temp,
                y_days_temp,
                y_flag_temp,
                meta_temp,
                test_size=0.25,  # 0.25 of remaining 80% => 20% overall
                random_state=13,
            )
            validation_strategy = "random_60_20_20_fallback"
            n_val = int(len(X_val))

        # --- baselines from the training window only (no leakage) ---
        self._global_mean_delay = float(y_days_train.mean())
        self._supplier_mean_delay = (
            meta_train.assign(_y=y_days_train.values)
            .groupby("supplier")["_y"]
            .mean()
            .to_dict()
        )
        self._global_late_rate = float(y_flag_train.mean())
        self._supplier_late_rate = (
            meta_train.assign(_y=y_flag_train.values)
            .groupby("supplier")["_y"]
            .mean()
            .to_dict()
        )

        baseline_days = meta_test["supplier"].map(self._supplier_mean_delay).fillna(self._global_mean_delay)
        baseline_mae = mean_absolute_error(y_days_test, baseline_days)

        baseline_prob = meta_test["supplier"].map(self._supplier_late_rate).fillna(self._global_late_rate)
        baseline_auc = _safe_auc(y_flag_test, baseline_prob)

        pos = float(y_flag_train.sum())
        neg = float(len(y_flag_train) - pos)
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        # Regressor: early stop on validation MAE.
        self.regressor = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            reg_lambda=1.0,
            random_state=13,
            early_stopping_rounds=40,
        )
        self.regressor.fit(
            X_train,
            y_days_train,
            eval_set=[(X_val, y_days_val)],
            verbose=False,
        )

        # Classifier: early stop on validation logloss; class weight for imbalance.
        self.classifier = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=13,
            eval_metric="logloss",
            early_stopping_rounds=40,
        )
        self.classifier.fit(
            X_train,
            y_flag_train,
            eval_set=[(X_val, y_flag_val)],
            verbose=False,
        )

        # Calibrate probabilities on validation (isotonic) — P enters $ decisions.
        raw_val = self._raw_proba(X_val)
        if y_flag_val.nunique() > 1 and len(y_flag_val) >= 30:
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw_val, y_flag_val.values)
        else:
            self._calibrator = None

        cal_val = self._calibrate_proba(raw_val)
        self.decision_threshold = _tune_threshold(y_flag_val, cal_val)

        # --- test metrics (untouched by tuning) ---
        pred_days = self.regressor.predict(X_test)
        mae = mean_absolute_error(y_days_test, pred_days)
        rmse = float(np.sqrt(mean_squared_error(y_days_test, pred_days)))
        r2 = float(r2_score(y_days_test, pred_days)) if y_days_test.nunique() > 1 else None

        raw_test = self._raw_proba(X_test)
        proba = self._calibrate_proba(raw_test)
        pred_flag = (proba >= self.decision_threshold).astype(int)
        auc = _safe_auc(y_flag_test, proba)
        pr_auc = _safe_pr_auc(y_flag_test, proba)

        precision = float(precision_score(y_flag_test, pred_flag, zero_division=0))
        recall = float(recall_score(y_flag_test, pred_flag, zero_division=0))
        f1 = float(f1_score(y_flag_test, pred_flag, zero_division=0))
        brier = float(brier_score_loss(y_flag_test, proba)) if y_flag_test.nunique() > 1 else None
        ece = _expected_calibration_error(y_flag_test, proba)
        cm = confusion_matrix(y_flag_test, pred_flag, labels=[0, 1]).tolist()
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        specificity = float(tn / (tn + fp)) if (tn + fp) else None
        npv = float(tn / (tn + fn)) if (tn + fn) else None
        false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) else None

        # Consistency: among predicted-late, mean predicted days should exceed threshold.
        late_mask = pred_flag == 1
        mean_days_when_pred_late = float(pred_days[late_mask].mean()) if late_mask.any() else None
        consistency_ok = (
            mean_days_when_pred_late is None
            or mean_days_when_pred_late >= DELAY_FLAG_THRESHOLD_DAYS * 0.75
        )

        # Segment diagnostics — useful when probability enters financial decisions.
        groups_supplier: dict[str, list[int]] = defaultdict(list)
        groups_region: dict[str, list[int]] = defaultdict(list)
        groups_peak: dict[str, list[int]] = defaultdict(list)
        meta_test_reset = meta_test.reset_index(drop=True)
        for i, row in meta_test_reset.iterrows():
            groups_supplier[row["supplier"]].append(i)
            groups_region[row["origin_region"]].append(i)
            groups_peak["peak" if row["is_peak_season"] else "off_peak"].append(i)

        y_flag_test_reset = y_flag_test.reset_index(drop=True)
        segment_auc = {
            "by_supplier": _group_metric(y_flag_test_reset, proba, pred_flag, groups_supplier, "auc"),
            "by_region": _group_metric(y_flag_test_reset, proba, pred_flag, groups_region, "auc"),
            "by_peak": _group_metric(y_flag_test_reset, proba, pred_flag, groups_peak, "auc"),
        }
        weak_segments = {
            scope: {k: v for k, v in vals.items() if v < 0.60}
            for scope, vals in segment_auc.items()
        }

        auc_lift_ci = _bootstrap_lift_ci(
            y_flag_test_reset.values, proba, baseline_prob.values, metric="auc"
        )
        mae_lift_ci = _bootstrap_lift_ci(
            y_days_test.reset_index(drop=True).values,
            pred_days,
            baseline_days.values,
            metric="mae",
        )

        best_clf_trees = int(getattr(self.classifier, "best_iteration", None) or self.classifier.n_estimators)
        best_reg_trees = int(getattr(self.regressor, "best_iteration", None) or self.regressor.n_estimators)

        metrics = {
            "mae_days": round(float(mae), 3),
            "rmse_days": round(rmse, 3),
            "r2_days": round(r2, 3) if r2 is not None else None,
            "auc": round(auc, 3) if auc is not None else None,
            "pr_auc": round(pr_auc, 3) if pr_auc is not None else None,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "specificity": round(specificity, 3) if specificity is not None else None,
            "npv": round(npv, 3) if npv is not None else None,
            "false_alarm_rate": round(false_alarm_rate, 3) if false_alarm_rate is not None else None,
            "brier_score": round(brier, 3) if brier is not None else None,
            "ece": round(ece, 3) if ece is not None else None,
            "confusion_matrix": cm,
            "decision_threshold": round(float(self.decision_threshold), 3),
            "scale_pos_weight": round(float(scale_pos_weight), 3),
            "baseline_mae_days": round(float(baseline_mae), 3),
            "baseline_auc": round(baseline_auc, 3) if baseline_auc is not None else None,
            "mae_lift_vs_baseline": round(float(baseline_mae - mae), 3),
            "auc_lift_vs_baseline": round(float((auc or 0) - (baseline_auc or 0)), 3)
            if auc is not None and baseline_auc is not None
            else None,
            "auc_lift_bootstrap": auc_lift_ci,
            "mae_lift_bootstrap": mae_lift_ci,
            "n_train": int(len(X_train)),
            "n_val": int(n_val),
            "n_test": int(len(X_test)),
            "validation_strategy": validation_strategy,
            "validation_used_for": [
                "early_stopping",
                "isotonic_calibration",
                "decision_threshold_tuning",
            ],
            "best_iteration_classifier": best_clf_trees,
            "best_iteration_regressor": best_reg_trees,
            "data_is_synthetic": True,
            "segment_auc": segment_auc,
            "weak_segments_auc_lt_0_60": weak_segments,
            "clf_reg_consistency_ok": bool(consistency_ok),
            "mean_pred_days_when_pred_late": (
                round(mean_days_when_pred_late, 3) if mean_days_when_pred_late is not None else None
            ),
            "probability_calibrated": self._calibrator is not None,
        }

        if verbose:
            print(
                f"[DelayModel] {validation_strategy}: train={metrics['n_train']} "
                f"val={metrics['n_val']} test={metrics['n_test']}"
            )
            print(
                f"  val used for: early stopping, isotonic calibration, "
                f"threshold={metrics['decision_threshold']}"
            )
            print(
                f"  regressor MAE {metrics['mae_days']}d RMSE {metrics['rmse_days']}d "
                f"R² {metrics['r2_days']} "
                f"(supplier-mean baseline {metrics['baseline_mae_days']}d, "
                f"lift {metrics['mae_lift_vs_baseline']}d; "
                f"bootstrap significant={mae_lift_ci.get('significant')})"
            )
            print(
                f"  classifier AUC {metrics['auc']} PR-AUC {metrics['pr_auc']} "
                f"(supplier-rate baseline {metrics['baseline_auc']}; "
                f"bootstrap significant={auc_lift_ci.get('significant')}) "
                f"P/R/F1 {metrics['precision']}/{metrics['recall']}/{metrics['f1']} "
                f"Brier {metrics['brier_score']} ECE {metrics['ece']}"
            )
            if any(weak_segments.values()):
                print(f"  WARN weak segments (AUC<0.60): {weak_segments}")
            print(
                "  note: metrics recover relationships programmed into synthetic data; "
                "they are not real-world supply-chain accuracy claims."
            )
        return metrics

    def feature_importance(self, top_n: int = 10) -> list[dict]:
        """Top features by average gain from the delay regressor (presentation-friendly)."""
        if self.regressor is None or self.feature_columns is None:
            raise RuntimeError("Model not trained/loaded yet")
        scores = self.regressor.feature_importances_
        ranked = sorted(
            zip(self.feature_columns, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_n]
        return [{"feature": name, "importance": round(float(score), 4)} for name, score in ranked]

    # -- inference ------------------------------------------------------
    def predict_one(self, shipment: dict) -> tuple[float, float]:
        """Returns (predicted_delay_days, predicted_delay_probability)."""
        if self.regressor is None or self.classifier is None:
            raise RuntimeError("Model not trained/loaded yet - call fit() or load()")

        row = pd.DataFrame([shipment])
        X, _ = build_features(row, categories=self.categories)
        # a category not seen at train time still needs to line up columns
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        predicted_days = float(self.regressor.predict(X)[0])
        raw = float(self._raw_proba(X)[0])
        predicted_prob = float(self._calibrate_proba(np.array([raw]))[0])
        return max(predicted_days, 0.0), predicted_prob

    # -- persistence ------------------------------------------------------
    def save(self, path) -> None:
        joblib.dump(
            {
                "classifier": self.classifier,
                "regressor": self.regressor,
                "categories": self.categories,
                "feature_columns": self.feature_columns,
                "supplier_mean_delay": self._supplier_mean_delay,
                "global_mean_delay": self._global_mean_delay,
                "supplier_late_rate": self._supplier_late_rate,
                "global_late_rate": self._global_late_rate,
                "decision_threshold": self.decision_threshold,
                "calibrator": self._calibrator,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "DelayModel":
        payload = joblib.load(path)
        model = cls()
        model.classifier = payload["classifier"]
        model.regressor = payload["regressor"]
        model.categories = payload["categories"]
        model.feature_columns = payload["feature_columns"]
        model._supplier_mean_delay = payload.get("supplier_mean_delay")
        model._global_mean_delay = payload.get("global_mean_delay", 0.0)
        model._supplier_late_rate = payload.get("supplier_late_rate")
        model._global_late_rate = payload.get("global_late_rate", 0.0)
        model.decision_threshold = float(payload.get("decision_threshold", 0.5))
        model._calibrator = payload.get("calibrator")
        return model
