"""
Week 1 - the predictive half of SupplyPrescript.

Two small XGBoost models sharing one feature set:
  - a classifier answering "will this ship late enough to matter?"
    (delay > features.DELAY_FLAG_THRESHOLD_DAYS)
  - a regressor answering "by roughly how many days?"

Both feed the Week 2 optimizer: expected holding cost uses
P(delay) × magnitude, so the classifier is part of the decision —
not a decorative metric.

Validation
----------
When `shipment_date` is present, fit() uses a temporal split
(60% train / 20% validation / 20% test):
train on earlier shipments, validate on the middle window, test on
the most recent period. That asks the business question correctly:
can historically observed behavior predict *future* outcomes?

Metrics are recovered from synthetic data with programmed
relationships (see generate_mock_data.py). An AUC of ~0.8 here means
"the model recovers the synthetic environment," not "we built a
0.8-AUC real-world delay model."
"""
from __future__ import annotations

from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
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
            # Fit on train only; report test metrics. Val is available for
            # threshold tuning / calibration work.
            X_train, X_test = X.loc[train_mask], X.loc[test_mask]
            y_days_train, y_days_test = y_days.loc[train_mask], y_days.loc[test_mask]
            y_flag_train, y_flag_test = y_flag.loc[train_mask], y_flag.loc[test_mask]
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
                _X_val,
                y_days_train,
                _y_days_val,
                y_flag_train,
                _y_flag_val,
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
            n_val = int(len(_X_val))

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

        self.regressor = XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=13,
        )
        self.regressor.fit(X_train, y_days_train)

        self.classifier = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=13,
            eval_metric="logloss",
        )
        self.classifier.fit(X_train, y_flag_train)

        pred_days = self.regressor.predict(X_test)
        mae = mean_absolute_error(y_days_test, pred_days)

        proba = self.classifier.predict_proba(X_test)[:, 1]
        pred_flag = (proba >= 0.5).astype(int)
        auc = _safe_auc(y_flag_test, proba)

        precision = float(precision_score(y_flag_test, pred_flag, zero_division=0))
        recall = float(recall_score(y_flag_test, pred_flag, zero_division=0))
        f1 = float(f1_score(y_flag_test, pred_flag, zero_division=0))
        brier = float(brier_score_loss(y_flag_test, proba)) if y_flag_test.nunique() > 1 else None
        cm = confusion_matrix(y_flag_test, pred_flag, labels=[0, 1]).tolist()

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
            "auc": round(auc, 3) if auc is not None else None,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "brier_score": round(brier, 3) if brier is not None else None,
            "confusion_matrix": cm,
            "baseline_mae_days": round(float(baseline_mae), 3),
            "baseline_auc": round(baseline_auc, 3) if baseline_auc is not None else None,
            "mae_lift_vs_baseline": round(float(baseline_mae - mae), 3),
            "n_train": int(len(X_train)),
            "n_val": int(n_val),
            "n_test": int(len(X_test)),
            "validation_strategy": validation_strategy,
            "data_is_synthetic": True,
            "segment_auc": segment_auc,
        }
        if verbose:
            print(
                f"[DelayModel] {validation_strategy}: train={metrics['n_train']} "
                f"val={metrics['n_val']} test={metrics['n_test']}"
            )
            print(
                f"  regressor MAE {metrics['mae_days']}d "
                f"(supplier-mean baseline {metrics['baseline_mae_days']}d, "
                f"lift {metrics['mae_lift_vs_baseline']}d)"
            )
            print(
                f"  classifier AUC {metrics['auc']} "
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
        predicted_prob = float(self.classifier.predict_proba(X)[0, 1])
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
        return model
