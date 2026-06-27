"""
Week 1 - the predictive half of SupplyPrescript.

Two small XGBoost models sharing one feature set:
  - a classifier answering "will this ship late enough to matter?"
    (delay > features.DELAY_FLAG_THRESHOLD_DAYS)
  - a regressor answering "by roughly how many days?"

Both feed the Week 2 optimizer: expected holding cost uses
P(delay) × magnitude, so the classifier is part of the decision —
not a decorative metric.

Capacity control
----------------
Trees are kept shallow with L2 / min-child / subsample regularization.
When a validation slice exists it is split in time: the earlier half
drives early stopping *and* a single over/under-fit adjustment; the
later half is reserved for probability calibration and F1 threshold
tuning so those steps do not reuse the early-stopping scores.

Validation
----------
When `shipment_date` is present, fit() uses a temporal split
(60% train / 20% validation / 20% test):
train on earlier shipments, use the middle window for stopping /
capacity / calibration, and report metrics only on the most recent
period. The test slice is never used to pick trees, depth, or threshold.

Metrics are recovered from synthetic data with programmed
relationships (see generate_mock_data.py). An AUC of ~0.8 here means
"the model recovers the synthetic environment," not "we built a
0.8-AUC real-world delay model."
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .features import build_features, DELAY_FLAG_THRESHOLD_DAYS

# Conservative defaults for ~thousands of shipment rows. Early stopping
# trims unused trees so n_estimators is a cap, not a target.
_BASE_BOOSTER = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 8,
    "gamma": 0.1,
    "reg_lambda": 2.0,
    "reg_alpha": 0.0,
    "random_state": 13,
    "tree_method": "hist",
    "n_jobs": 1,
}
_EARLY_STOPPING_ROUNDS = 40
# Relative (val - train) / val above this ⇒ overfit; one regularization bump.
_OVERFIT_REL_GAP = 0.25
# Val MAE must beat the supplier-mean baseline by this fraction or we add capacity.
_UNDERFIT_VS_BASELINE = 0.98


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


def _expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float | None:
    """ECE — gap between predicted P(delay) and observed late frequency."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0 or pd.Series(y_true).nunique() < 2:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / total) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def _tune_threshold(y_true, y_prob, max_fpr: float = 0.35) -> float:
    """Pick a calibration-slice threshold under an FPR ceiling, then maximize F1.

    Unchecked false alarms burn air-freight budget; missed lates (FN) still
    matter, so we take the best F1 among thresholds with FPR ≤ max_fpr.
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
        score = float(f1_score(y_true, pred, zero_division=0))
        candidates.append((float(t), score, fpr, tp))
    feasible = [c for c in candidates if c[2] <= max_fpr and c[3] > 0]
    if feasible:
        return max(feasible, key=lambda c: c[1])[0]
    with_tp = [c for c in candidates if c[3] > 0]
    if not with_tp:
        return 0.5
    return min(with_tp, key=lambda c: (c[2], -c[1]))[0]


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


def _align_xy(frame: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    return frame.reset_index(drop=True), target.reset_index(drop=True)


def split_val_stop_tune(
    X_val: pd.DataFrame,
    y_days_val: pd.Series,
    y_flag_val: pd.Series,
    meta_val: pd.DataFrame,
    dates: pd.Series | None = None,
    min_each: int = 20,
) -> dict:
    """Split validation into an early-stopping slice and a calibration slice.

    Temporal: earlier val → stop, later val → tune.
    Random/no dates: second half after a stratified shuffle.
    If the slice is too small, both roles share the same rows.
    """
    X_val, y_days_val = _align_xy(X_val, y_days_val)
    y_flag_val = y_flag_val.reset_index(drop=True)
    meta_val = meta_val.reset_index(drop=True)
    n = len(X_val)
    if n < min_each * 2:
        shared = {
            "X_stop": X_val,
            "y_days_stop": y_days_val,
            "y_flag_stop": y_flag_val,
            "meta_stop": meta_val,
            "X_tune": X_val,
            "y_days_tune": y_days_val,
            "y_flag_tune": y_flag_val,
            "shared": True,
        }
        return shared

    if dates is not None:
        order = pd.to_datetime(dates.reset_index(drop=True)).argsort(kind="mergesort").to_numpy()
        cut = n // 2
        stop_idx, tune_idx = order[:cut], order[cut:]
    else:
        idx = np.arange(n)
        strat = y_flag_val if int(y_flag_val.nunique()) > 1 else None
        stop_idx, tune_idx = train_test_split(
            idx, test_size=0.5, random_state=13, stratify=strat
        )

    def _take(i):
        return {
            "X": X_val.iloc[i],
            "y_days": y_days_val.iloc[i],
            "y_flag": y_flag_val.iloc[i],
            "meta": meta_val.iloc[i],
        }

    stop, tune = _take(stop_idx), _take(tune_idx)
    return {
        "X_stop": stop["X"],
        "y_days_stop": stop["y_days"],
        "y_flag_stop": stop["y_flag"],
        "meta_stop": stop["meta"],
        "X_tune": tune["X"],
        "y_days_tune": tune["y_days"],
        "y_flag_tune": tune["y_flag"],
        "shared": False,
    }


def diagnose_capacity(
    train_mae: float,
    val_mae: float,
    baseline_val_mae: float,
    train_baseline_mae: float,
) -> str:
    """Label capacity using baselines so a temporal shift is not called overfit.

    Overfit: fits the training window but does not beat the val baseline.
    Underfit: fails to beat the supplier-mean baseline on *both* train and val.
    Balanced: generalizes at least as well as the naive supplier mean on val.
    """
    rel_gap = (float(val_mae) - float(train_mae)) / max(float(val_mae), 1e-6)
    fits_train = float(train_mae) < float(train_baseline_mae) * _UNDERFIT_VS_BASELINE
    beats_val_baseline = float(val_mae) < float(baseline_val_mae) * _UNDERFIT_VS_BASELINE
    if fits_train and (not beats_val_baseline) and rel_gap > _OVERFIT_REL_GAP:
        return "overfit"
    if (not fits_train) and (not beats_val_baseline):
        return "underfit"
    return "balanced"


def adjust_booster_params(params: dict, diagnosis: str) -> dict | None:
    """One conservative capacity step. None means keep the current trees."""
    if diagnosis == "overfit":
        nxt = deepcopy(params)
        nxt["max_depth"] = max(2, int(params["max_depth"]) - 1)
        nxt["min_child_weight"] = float(params["min_child_weight"]) * 1.5
        nxt["reg_lambda"] = float(params["reg_lambda"]) * 2.0
        nxt["subsample"] = min(float(params["subsample"]), 0.7)
        nxt["colsample_bytree"] = min(float(params["colsample_bytree"]), 0.7)
        return nxt
    if diagnosis == "underfit":
        nxt = deepcopy(params)
        nxt["max_depth"] = min(5, int(params["max_depth"]) + 1)
        nxt["min_child_weight"] = max(1.0, float(params["min_child_weight"]) / 2.0)
        nxt["reg_lambda"] = max(1.0, float(params["reg_lambda"]) / 2.0)
        return nxt
    return None


def _best_trees(estimator) -> int | None:
    iteration = getattr(estimator, "best_iteration", None)
    if iteration is None:
        return None
    return int(iteration) + 1


def _scale_pos_weight(y_flag: pd.Series) -> float:
    pos = float((y_flag == 1).sum())
    neg = float((y_flag == 0).sum())
    if pos <= 0:
        return 1.0
    return neg / pos


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
        self.decision_threshold: float = 0.5
        self._calibrator: IsotonicRegression | None = None
        self.fit_quality: str | None = None
        self.booster_params: dict | None = None

    def _raw_proba(self, X) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Classifier not trained")
        return self.classifier.predict_proba(X)[:, 1]

    def _calibrate_proba(self, raw_proba) -> np.ndarray:
        raw = np.asarray(raw_proba, dtype=float)
        if self._calibrator is None:
            return np.clip(raw, 0.0, 1.0)
        return np.clip(self._calibrator.predict(raw), 0.0, 1.0)

    def _fit_boosters(
        self,
        params: dict,
        X_train,
        y_days_train,
        y_flag_train,
        eval_reg,
        eval_clf,
        use_val: bool,
        scale_pos_weight: float,
    ) -> None:
        stop_kw = {"early_stopping_rounds": _EARLY_STOPPING_ROUNDS} if use_val else {}
        self.regressor = XGBRegressor(**params, **stop_kw)
        if use_val:
            self.regressor.fit(X_train, y_days_train, eval_set=eval_reg, verbose=False)
        else:
            self.regressor.fit(X_train, y_days_train)

        clf_params = {**params, "eval_metric": "logloss", "scale_pos_weight": scale_pos_weight}
        self.classifier = XGBClassifier(**clf_params, **stop_kw)
        if use_val:
            self.classifier.fit(X_train, y_flag_train, eval_set=eval_clf, verbose=False)
        else:
            self.classifier.fit(X_train, y_flag_train)
        self.booster_params = dict(params)

    # -- training -----------------------------------------------------
    def fit(self, shipments: pd.DataFrame, verbose: bool = True) -> dict:
        shipments = shipments.reset_index(drop=True)
        X, self.categories = build_features(shipments)
        self.feature_columns = X.columns.tolist()
        y_days = shipments["actual_delay_days"].reset_index(drop=True)
        y_flag = (y_days > DELAY_FLAG_THRESHOLD_DAYS).astype(int)

        has_dates = "shipment_date" in shipments.columns
        val_dates = None
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
            meta_val = shipments.loc[val_mask]
            meta_test = shipments.loc[test_mask]
            val_dates = dates.loc[val_mask]
            validation_strategy = "temporal_60_20_20"
            n_val = int(val_mask.sum())
        else:
            # Random 60:20:20 fallback when dates are missing.
            strat_all = y_flag if int(y_flag.nunique()) > 1 else None
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
                X,
                y_days,
                y_flag,
                shipments,
                test_size=0.20,
                random_state=13,
                stratify=strat_all,
            )
            strat_temp = y_flag_temp if int(pd.Series(y_flag_temp).nunique()) > 1 else None
            (
                X_train,
                X_val,
                y_days_train,
                y_days_val,
                y_flag_train,
                y_flag_val,
                meta_train,
                meta_val,
            ) = train_test_split(
                X_temp,
                y_days_temp,
                y_flag_temp,
                meta_temp,
                test_size=0.25,  # 0.25 of remaining 80% => 20% overall
                random_state=13,
                stratify=strat_temp,
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

        slices = split_val_stop_tune(
            X_val, y_days_val, y_flag_val, meta_val, dates=val_dates
        )
        X_stop, y_days_stop, y_flag_stop, meta_stop = (
            slices["X_stop"],
            slices["y_days_stop"],
            slices["y_flag_stop"],
            slices["meta_stop"],
        )
        X_tune, y_flag_tune = slices["X_tune"], slices["y_flag_tune"]
        n_val_stop, n_val_tune = int(len(X_stop)), int(len(X_tune))

        baseline_val_days = meta_stop["supplier"].map(self._supplier_mean_delay).fillna(self._global_mean_delay)
        baseline_val_mae = float(mean_absolute_error(y_days_stop, baseline_val_days))
        train_base_days = meta_train["supplier"].map(self._supplier_mean_delay).fillna(self._global_mean_delay)
        train_baseline_mae = float(mean_absolute_error(y_days_train, train_base_days))

        use_val = n_val_stop >= 20 and int(pd.Series(y_flag_stop).nunique()) > 1
        scale_pos = _scale_pos_weight(y_flag_train.reset_index(drop=True))
        params = deepcopy(_BASE_BOOSTER)
        eval_reg = [(X_stop, y_days_stop)] if use_val else None
        eval_clf = [(X_stop, y_flag_stop)] if use_val else None

        self._fit_boosters(
            params,
            X_train,
            y_days_train,
            y_flag_train,
            eval_reg,
            eval_clf,
            use_val,
            scale_pos,
        )

        capacity_adjustment = None
        if use_val:
            train_mae_probe = float(mean_absolute_error(y_days_train, self.regressor.predict(X_train)))
            val_mae_probe = float(mean_absolute_error(y_days_stop, self.regressor.predict(X_stop)))
            diagnosis = diagnose_capacity(
                train_mae_probe, val_mae_probe, baseline_val_mae, train_baseline_mae
            )
            nxt = adjust_booster_params(params, diagnosis)
            if nxt is not None:
                capacity_adjustment = diagnosis
                self._fit_boosters(
                    nxt,
                    X_train,
                    y_days_train,
                    y_flag_train,
                    eval_reg,
                    eval_clf,
                    use_val,
                    scale_pos,
                )
                train_mae_probe = float(mean_absolute_error(y_days_train, self.regressor.predict(X_train)))
                val_mae_probe = float(mean_absolute_error(y_days_stop, self.regressor.predict(X_stop)))
                diagnosis = diagnose_capacity(
                    train_mae_probe, val_mae_probe, baseline_val_mae, train_baseline_mae
                )
            self.fit_quality = diagnosis
        else:
            train_mae_probe = float(mean_absolute_error(y_days_train, self.regressor.predict(X_train)))
            val_mae_probe = None
            self.fit_quality = "unvalidated"

        if use_val:
            raw_tune = self._raw_proba(X_tune)
            self._calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._calibrator.fit(raw_tune, np.asarray(y_flag_tune, dtype=float))
            self.decision_threshold = _tune_threshold(y_flag_tune, self._calibrate_proba(raw_tune))
        else:
            self._calibrator = None
            self.decision_threshold = 0.5

        pred_days = self.regressor.predict(X_test)
        mae = mean_absolute_error(y_days_test, pred_days)

        proba = self._calibrate_proba(self._raw_proba(X_test))
        pred_flag = (proba >= self.decision_threshold).astype(int)
        auc = _safe_auc(y_flag_test, proba)

        precision = float(precision_score(y_flag_test, pred_flag, zero_division=0))
        recall = float(recall_score(y_flag_test, pred_flag, zero_division=0))
        f1 = float(f1_score(y_flag_test, pred_flag, zero_division=0))
        brier = float(brier_score_loss(y_flag_test, proba)) if y_flag_test.nunique() > 1 else None
        ece = _expected_calibration_error(y_flag_test, proba)
        cm = confusion_matrix(y_flag_test, pred_flag, labels=[0, 1]).tolist()

        auc_train = _safe_auc(y_flag_train, self._calibrate_proba(self._raw_proba(X_train)))
        auc_val = (
            _safe_auc(y_flag_stop, self._calibrate_proba(self._raw_proba(X_stop)))
            if use_val
            else None
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

        metrics = {
            "mae_days": round(float(mae), 3),
            "mae_train": round(float(train_mae_probe), 3),
            "mae_val": round(float(val_mae_probe), 3) if val_mae_probe is not None else None,
            "baseline_val_mae_days": round(float(baseline_val_mae), 3),
            "auc": round(auc, 3) if auc is not None else None,
            "auc_train": round(auc_train, 3) if auc_train is not None else None,
            "auc_val": round(auc_val, 3) if auc_val is not None else None,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "brier_score": round(brier, 3) if brier is not None else None,
            "ece": round(ece, 3) if ece is not None else None,
            "confusion_matrix": cm,
            "baseline_mae_days": round(float(baseline_mae), 3),
            "baseline_auc": round(baseline_auc, 3) if baseline_auc is not None else None,
            "mae_lift_vs_baseline": round(float(baseline_mae - mae), 3),
            "n_train": int(len(X_train)),
            "n_val": int(n_val),
            "n_val_stop": n_val_stop,
            "n_val_tune": n_val_tune,
            "n_test": int(len(X_test)),
            "validation_strategy": validation_strategy,
            "validation_used_for_tuning": bool(use_val),
            "val_stop_tune_shared": bool(slices["shared"]),
            "decision_threshold": round(float(self.decision_threshold), 3),
            "data_is_synthetic": True,
            "segment_auc": segment_auc,
            "fit_quality": self.fit_quality,
            "capacity_adjustment": capacity_adjustment,
            "n_estimators_cap": int(self.booster_params["n_estimators"]) if self.booster_params else None,
            "n_estimators_regressor": _best_trees(self.regressor),
            "n_estimators_classifier": _best_trees(self.classifier),
            "max_depth": int(self.booster_params["max_depth"]) if self.booster_params else None,
            "scale_pos_weight": round(float(scale_pos), 3),
        }
        if verbose:
            print(
                f"[DelayModel] {validation_strategy}: train={metrics['n_train']} "
                f"val={metrics['n_val']} (stop={n_val_stop} tune={n_val_tune}) "
                f"test={metrics['n_test']}"
            )
            print(
                f"  regressor MAE train {metrics['mae_train']}d / val {metrics['mae_val']}d / "
                f"test {metrics['mae_days']}d "
                f"(supplier-mean baseline {metrics['baseline_mae_days']}d, "
                f"lift {metrics['mae_lift_vs_baseline']}d) "
                f"trees={metrics['n_estimators_regressor']}/{metrics['n_estimators_cap']} "
                f"depth={metrics['max_depth']} fit={metrics['fit_quality']}"
            )
            print(
                f"  classifier AUC train {metrics['auc_train']} / val {metrics['auc_val']} / "
                f"test {metrics['auc']} "
                f"(supplier-rate baseline {metrics['baseline_auc']}) "
                f"P/R/F1 {metrics['precision']}/{metrics['recall']}/{metrics['f1']} "
                f"Brier {metrics['brier_score']}"
            )
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
        predicted_prob = float(self._calibrate_proba(self._raw_proba(X))[0])
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
                "fit_quality": self.fit_quality,
                "booster_params": self.booster_params,
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
        model.fit_quality = payload.get("fit_quality")
        model.booster_params = payload.get("booster_params")
        return model
