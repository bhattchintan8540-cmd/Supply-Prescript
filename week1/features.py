"""
Turns a shipment (raw dict / dataframe row) into the numeric feature
vector the model expects. Kept separate from delay_model.py so training
and live inference can't quietly drift apart from each other - both
paths call build_features().
"""
from __future__ import annotations

import pandas as pd

CATEGORICAL_COLS = ["supplier", "origin_region", "sku"]
NUMERIC_COLS = [
    "distance_km",
    "historical_avg_lead_time_days",
    "order_quantity",
    "unit_cost_usd",
    "is_peak_season",
]

# Delay is "significant" past this many days - used to derive the binary
# label for the probability model from the regression target.
DELAY_FLAG_THRESHOLD_DAYS = 3.0

_PEAK_TRUE = {"true", "t", "yes", "y", "1", "peak"}
_PEAK_FALSE = {"false", "f", "no", "n", "0", "off", "off-peak", "off_peak", ""}


def coerce_peak_season(series: pd.Series) -> pd.Series:
    """Map mixed peak-season encodings to 0/1 without crashing.

    CSV dumps, Excel, and manual rows often store this as 'True'/'False'
    (or yes/no). pandas `.astype(int)` only works for bool/numeric dtypes
    and raises ValueError on those strings.
    """
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)

    def _one(value) -> int:
        if value is True or value is False:
            return int(value)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0
        text = str(value).strip().lower()
        if text in _PEAK_TRUE:
            return 1
        if text in _PEAK_FALSE:
            return 0
        try:
            return int(float(text) != 0.0)
        except (TypeError, ValueError):
            return 0

    return series.map(_one).astype(int)


def build_features(df: pd.DataFrame, categories: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One-hot encode the categorical columns and pass numerics through.

    `categories` lets inference reuse the exact column set training saw
    (a category that only showed up once at train time still needs a
    column at predict time, even if it's all zeros). If not given, the
    categories present in `df` are used and returned so callers can
    persist them alongside the trained model.
    """
    df = df.copy()
    df["is_peak_season"] = coerce_peak_season(df["is_peak_season"])

    if categories is None:
        categories = {col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_COLS}

    encoded_blocks = [df[NUMERIC_COLS].reset_index(drop=True)]
    for col in CATEGORICAL_COLS:
        cat_dtype = pd.CategoricalDtype(categories=categories[col])
        dummies = pd.get_dummies(df[col].astype(cat_dtype), prefix=col)
        encoded_blocks.append(dummies.reset_index(drop=True))

    features = pd.concat(encoded_blocks, axis=1)
    return features, categories
