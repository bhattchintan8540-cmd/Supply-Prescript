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
    assert "month_sin" in features.columns and "month_cos" in features.columns


def test_cyclic_month_uses_shipment_date_when_present():
    dec = {**_row(False), "shipment_date": "2024-12-15"}
    jun = {**_row(False), "shipment_date": "2024-06-15"}
    df = pd.DataFrame([dec, jun])
    features, _ = build_features(df)
    # December (angle 2π) and June (angle π) sit on opposite sides of the circle.
    assert features.loc[0, "month_cos"] > 0.9
    assert features.loc[1, "month_cos"] < -0.9
    assert abs(features.loc[0, "month_sin"]) < 0.1
    assert abs(features.loc[1, "month_sin"]) < 0.1



def test_coerce_peak_season_accepts_yes_no_and_numeric():
    series = pd.Series(["yes", "NO", 1, 0, True, False, None])
    assert list(coerce_peak_season(series)) == [1, 0, 1, 0, 1, 0, 0]


def test_seed_shipments_table_writes_orm_rows():
    from week1 import models
    from week1.database import SessionLocal
    from week1.generate_mock_data import build, seed_shipments_table

    df = build(n_rows=25)
    assert seed_shipments_table(df) == 25
    session = SessionLocal()
    try:
        assert session.query(models.Shipment).count() == 25
    finally:
        session.close()
