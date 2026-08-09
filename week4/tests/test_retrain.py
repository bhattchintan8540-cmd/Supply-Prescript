import json

from week1 import models
from week1.database import SessionLocal
from week4.retrain import average_cost_drift, maybe_retrain, outcomes_as_training_rows


def _clear_decisions() -> None:
    """Week 3 API tests write resolved decisions into the shared test DB.
    Clear them so Week 4 assertions about an empty loop stay true."""
    session = SessionLocal()
    try:
        session.query(models.Decision).delete()
        session.commit()
    finally:
        session.close()


def test_average_cost_drift_is_none_with_no_resolved_decisions():
    _clear_decisions()
    session = SessionLocal()
    try:
        drift = average_cost_drift(session)
    finally:
        session.close()
    assert drift is None


def test_maybe_retrain_skips_when_nothing_resolved_yet():
    _clear_decisions()
    result = maybe_retrain()
    assert result["retrained"] is False
    assert result["drift"] is None


def test_outcomes_as_training_rows_uses_feature_snapshots():
    _clear_decisions()
    session = SessionLocal()
    try:
        features = {
            "sku": "MICROCHIP-A2",
            "supplier": "NovaChip Manufacturing",
            "origin_region": "Asia Pacific",
            "distance_km": 8800,
            "historical_avg_lead_time_days": 16,
            "order_quantity": 6000,
            "unit_cost_usd": 14.2,
            "is_peak_season": False,
        }
        decision = models.Decision(
            shipment_sku="MICROCHIP-A2",
            predicted_delay_days=4.0,
            predicted_delay_probability=0.6,
            options_json="[]",
            shipment_features_json=json.dumps(features),
            chosen_option_label="Air Freight",
            predicted_cost_usd=90000,
            no_action_cost_usd=95000,
            budget_cap_usd=100000,
            actual_cost_usd=91000,
            actual_delay_days=1.5,
        )
        session.add(decision)
        session.commit()

        rows = outcomes_as_training_rows(session)
        assert len(rows) == 1
        assert rows.iloc[0]["actual_delay_days"] == 1.5
        assert rows.iloc[0]["supplier"] == "NovaChip Manufacturing"
    finally:
        session.close()
