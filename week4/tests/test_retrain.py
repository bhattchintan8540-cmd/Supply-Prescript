from week1 import models
from week1.database import SessionLocal
from week4.retrain import average_cost_drift, maybe_retrain


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
