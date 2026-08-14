import pandas as pd

from week1.features import build_features, coerce_peak_season


def _row(peak):
    return {
        "supplier": "A",
        "origin_region": "Europe",
        "sku": "X",
        "distance_km": 1,
        "historical_avg_lead_time_days": 1,
        "order_quantity": 1,
        "unit_cost_usd": 1.0,
        "is_peak_season": peak,
    }


def test_string_true_false_peak_flags_do_not_crash():
    """CSV / Excel often store booleans as the strings 'True'/'False'."""
    df = pd.DataFrame([_row("True"), _row("False")])
    features, _ = build_features(df)
    assert list(features["is_peak_season"]) == [1, 0]


def test_coerce_peak_season_accepts_yes_no_and_numeric():
    series = pd.Series(["yes", "NO", 1, 0, True, False, None])
    assert list(coerce_peak_season(series)) == [1, 0, 1, 0, 1, 0, 0]
