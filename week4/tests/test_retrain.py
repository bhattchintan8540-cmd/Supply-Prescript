from week4.retrain import average_cost_drift, maybe_retrain
from week1.database import SessionLocal


def test_average_cost_drift_is_none_with_no_resolved_decisions():
    session = SessionLocal()
    try:
        drift = average_cost_drift(session)
    finally:
        session.close()
    assert drift is None


def test_maybe_retrain_skips_when_nothing_resolved_yet():
    result = maybe_retrain()
    assert result["retrained"] is False
    assert result["drift"] is None
