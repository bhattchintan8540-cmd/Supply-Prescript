"""Phase 3 drift gates: retrain only when a signal is over its limit."""

import json
from datetime import datetime, timezone

from week1 import models
from week1.database import SessionLocal
from week4.retrain import outcome_drift_signals, outcomes_as_training_rows, phase3_status


def _clear_decisions() -> None:
    session = SessionLocal()
    try:
        session.query(models.Decision).delete()
        session.commit()
    finally:
        session.close()


def test_phase3_status_is_quiet_with_no_resolved_outcomes():
    _clear_decisions()
    session = SessionLocal()
    try:
        status = phase3_status(session)
    finally:
        session.close()
    assert status["resolved_decisions"] == 0
    assert status["should_retrain"] is False
    assert status["triggers"] == []
    assert status["thresholds"]["cost_mape"] == 0.15


def test_cost_error_over_fifteen_percent_requests_a_retrain():
    _clear_decisions()
    session = SessionLocal()
    try:
        session.add(
            models.Decision(
                shipment_sku="MICROCHIP-A2",
                predicted_delay_days=6.0,
                predicted_delay_probability=0.8,
                options_json="[]",
                chosen_option_label="Secondary Supplier",
                predicted_cost_usd=90000,
                no_action_cost_usd=100000,
                budget_cap_usd=100000,
                actual_cost_usd=120000,
                actual_delay_days=6.0,
                created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                resolved_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
            )
        )
        session.commit()
        signals = outcome_drift_signals(session)
        status = phase3_status(session)
        assert signals is not None
        assert signals["cost_mape"] >= 0.15
        assert "cost_mape" in signals["triggers"]
        assert status["should_retrain"] is True
        assert outcomes_as_training_rows(session).empty
    finally:
        session.close()
