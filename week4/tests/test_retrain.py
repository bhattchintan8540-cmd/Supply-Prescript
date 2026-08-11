import json
from datetime import datetime, timezone

import pytest

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
            # Decision-time date (not resolution time) drives temporal split.
            created_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
            resolved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )
        session.add(decision)
        session.commit()

        rows = outcomes_as_training_rows(session)
        assert len(rows) == 1
        assert rows.iloc[0]["actual_delay_days"] == 1.5
        assert rows.iloc[0]["supplier"] == "NovaChip Manufacturing"
        assert rows.iloc[0]["shipment_date"] == "2026-07-15"
    finally:
        session.close()


def test_should_retrain_triggers_on_hard_miss_even_if_cost_ok():
    from week4.retrain import should_retrain

    diagnostics = {
        "cost_mape": 0.05,  # under 15%
        "hard_miss_rate": 0.45,
        "outcome_brier": 0.12,
        "delay_mae_days": 1.0,
    }
    do_it, reason = should_retrain(diagnostics)
    assert do_it is True
    assert "hard miss" in reason


def test_prediction_drift_diagnostics_reports_signals():
    from week4.retrain import prediction_drift_diagnostics

    _clear_decisions()
    session = SessionLocal()
    try:
        session.add(
            models.Decision(
                shipment_sku="MICROCHIP-A2",
                predicted_delay_days=2.0,
                predicted_delay_probability=0.1,
                options_json="[]",
                chosen_option_label="Delay Launch",
                predicted_cost_usd=10000,
                no_action_cost_usd=10000,
                budget_cap_usd=100000,
                actual_cost_usd=11000,
                actual_delay_days=8.0,  # late, but model was confident on-time
                resolved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            )
        )
        session.commit()
        diag = prediction_drift_diagnostics(session)
        assert diag["n_resolved"] == 1
        assert diag["cost_mape"] == pytest.approx(0.1)
        assert diag["delay_mae_days"] == pytest.approx(6.0)
        assert diag["hard_miss_rate"] == 1.0
        assert diag["outcome_brier"] is not None
    finally:
        session.close()


def test_outcomes_without_resolved_at_are_skipped():
    _clear_decisions()
    session = SessionLocal()
    try:
        features = {
            "sku": "SENSOR-IR",
            "supplier": "Meridian Fasteners",
            "origin_region": "North America",
            "distance_km": 2400,
            "historical_avg_lead_time_days": 9,
            "order_quantity": 4000,
            "unit_cost_usd": 8.5,
            "is_peak_season": False,
        }
        decision = models.Decision(
            shipment_sku="SENSOR-IR",
            predicted_delay_days=2.0,
            predicted_delay_probability=0.3,
            options_json="[]",
            shipment_features_json=json.dumps(features),
            chosen_option_label="Delay Launch",
            predicted_cost_usd=40000,
            no_action_cost_usd=40000,
            budget_cap_usd=100000,
            actual_cost_usd=41000,
            actual_delay_days=2.0,
            resolved_at=None,
        )
        session.add(decision)
        session.commit()
        rows = outcomes_as_training_rows(session)
        assert len(rows) == 0
    finally:
        session.close()
