"""Older local sqlite files gain columns that create_all will not add."""

from sqlalchemy import create_engine, inspect, text

from week1.database import ensure_missing_columns


def test_old_decisions_table_gains_feature_snapshot(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE decisions (
                    id INTEGER PRIMARY KEY,
                    shipment_sku VARCHAR NOT NULL,
                    predicted_delay_days FLOAT NOT NULL,
                    predicted_delay_probability FLOAT NOT NULL,
                    options_json TEXT NOT NULL,
                    chosen_option_label VARCHAR NOT NULL,
                    predicted_cost_usd FLOAT NOT NULL,
                    budget_cap_usd FLOAT NOT NULL,
                    actual_cost_usd FLOAT,
                    actual_delay_days FLOAT,
                    created_at DATETIME
                )
                """
            )
        )
    ensure_missing_columns(engine)
    columns = {col["name"] for col in inspect(engine).get_columns("decisions")}
    assert "shipment_features_json" in columns
    assert "no_action_cost_usd" in columns
    assert "resolved_at" in columns
