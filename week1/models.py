from __future__ import annotations

import datetime as dt

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text

from .database import Base


class Shipment(Base):
    """One historical shipment record - this is what the delay model
    trains on. Prefer real open data via week1/ingest_real_data.py
    (USAID SCMS / UCI Cargo 2000); mock rows remain available from
    week1/generate_mock_data.py for offline demos."""

    __tablename__ = "shipments"

    id = Column(Integer, primary_key=True)
    sku = Column(String, nullable=False)
    supplier = Column(String, nullable=False)
    origin_region = Column(String, nullable=False)
    distance_km = Column(Float, nullable=False)
    historical_avg_lead_time_days = Column(Float, nullable=False)
    order_quantity = Column(Integer, nullable=False)
    unit_cost_usd = Column(Float, nullable=False)
    is_peak_season = Column(Boolean, default=False)
    # label the model is trained against
    actual_delay_days = Column(Float, nullable=False)
    # provenance: "usaid-scms", "uci-c2k", "mock", …
    data_source = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.UTC))


class Decision(Base):
    """A single closed-loop record: prediction -> prescribed options ->
    the option a human picked -> (eventually) what actually happened.

    Rows start with actual_cost_usd / actual_delay_days as NULL and get
    backfilled once the real-world outcome is known - that's the "close
    the loop" step described in the project doc.
    """

    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True)
    shipment_sku = Column(String, nullable=False)
    predicted_delay_days = Column(Float, nullable=False)
    predicted_delay_probability = Column(Float, nullable=False)

    # snapshot of the three prescribed options at decision time, stored as
    # JSON text - keeps the schema simple and the audit trail intact even
    # if the solver's exact option set changes later.
    options_json = Column(Text, nullable=False)

    chosen_option_label = Column(String, nullable=False)
    predicted_cost_usd = Column(Float, nullable=False)
    budget_cap_usd = Column(Float, nullable=False)

    actual_cost_usd = Column(Float, nullable=True)
    actual_delay_days = Column(Float, nullable=True)

    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.UTC))
    resolved_at = Column(DateTime, nullable=True)

    @property
    def is_resolved(self) -> bool:
        return self.actual_cost_usd is not None
