"""
Week 1 - the predictive half of SupplyPrescript.

Two small XGBoost models sharing one feature set:
  - a classifier answering "will this ship late enough to matter?"
    (delay > features.DELAY_FLAG_THRESHOLD_DAYS)
  - a regressor answering "by roughly how many days?"

Splitting probability from magnitude like this is a bit more work than
a single regressor, but the solver needs both numbers separately -
"85% chance of a moderate delay" and "guaranteed 1-day slip" call for
very different prescriptions even if their expected value is similar.
"""
from __future__ import annotations

import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score

from .features import build_features, DELAY_FLAG_THRESHOLD_DAYS


class DelayModel:
    def __init__(self):
        self.classifier: XGBClassifier | None = None
        self.regressor: XGBRegressor | None = None
        self.categories: dict[str, list[str]] | None = None
        self.feature_columns: list[str] | None = None

    # -- training -----------------------------------------------------
    def fit(self, shipments: pd.DataFrame, verbose: bool = True) -> dict:
        X, self.categories = build_features(shipments)
        self.feature_columns = X.columns.tolist()
        y_days = shipments["actual_delay_days"].reset_index(drop=True)
        y_flag = (y_days > DELAY_FLAG_THRESHOLD_DAYS).astype(int)

        X_train, X_test, y_days_train, y_days_test, y_flag_train, y_flag_test = train_test_split(
            X, y_days, y_flag, test_size=0.2, random_state=13
        )

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

        mae = mean_absolute_error(y_days_test, self.regressor.predict(X_test))
        # roc_auc needs both classes present - guard the mock-data edge case
        auc = roc_auc_score(y_flag_test, self.classifier.predict_proba(X_test)[:, 1]) if y_flag_test.nunique() > 1 else float("nan")

        metrics = {"mae_days": round(mae, 3), "auc": round(auc, 3) if auc == auc else None, "n_train": len(X_train), "n_test": len(X_test)}
        if verbose:
            print(f"[DelayModel] trained on {metrics['n_train']} rows, "
                  f"held out {metrics['n_test']} - MAE {metrics['mae_days']}d, AUC {metrics['auc']}")
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
        return model
