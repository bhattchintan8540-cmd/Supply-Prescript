"""Tests for real open-data adapters (USAID SCMS + UCI Cargo 2000)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from week1.data_adapters import SHIPMENT_COLUMNS, transform_c2k, transform_scms

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def scms_sample(tmp_path_factory) -> Path:
    """Tiny SCMS-shaped CSV so tests never need the network."""
    path = tmp_path_factory.mktemp("scms") / "sample.csv"
    path.write_text(
        "\n".join(
            [
                "ID,Country,Vendor,Manufacturing Site,Shipment Mode,Product Group,Brand,"
                "Molecule/Test Type,Item Description,Line Item Quantity,Line Item Value,"
                "Pack_Price,Unit_Price,Weight,Freight_Cost,"
                "Scheduled Delivery Date,Delivered to Client Date,PO Sent to Vendor Date",
                "1,Nigeria,Aurobindo Pharma Limited,\"Aurobindo Unit III, India\",Ocean,ARV,Generic,"
                "Lamivudine,Lamivudine 150mg,10000,1800,5.4,0.18,1200,800,"
                "15-Jan-14,20-Jan-14,1/2/13",
                "2,Uganda,Orgenics Ltd,Orgenics Israel,Air,HRDT,Determine,"
                "HIV Determine,HIV Determine Test,500,700,1.4,1.4,40,220,"
                "10-Dec-14,25-Dec-14,Date Not Captured",
                "3,Vietnam,Trinity Biotech Plc,\"Trinity Biotech, Plc\",Air,HRDT,Uni-Gold,"
                "HIV Uni-Gold,HIV Uni-Gold Test,200,400,2.0,2.0,15,90,"
                "1-Jun-13,1-Jun-13,3/1/13",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def c2k_sample(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("c2k") / "c2k.csv"
    # Minimal Cargo-2000 shaped row: planned/effective delivery minutes + place.
    path.write_text(
        "nr,o_dlv_p,o_dlv_e,o_hops,legs,o_dep_1_place\n"
        "0,1440,2880,1,2,700\n"
        "1,2880,2880,2,1,128\n"
        "11,1000,5000,1,3,815\n",
        encoding="utf-8",
    )
    return path


def test_transform_scms_maps_required_columns(scms_sample):
    df = transform_scms(scms_sample)
    assert list(df.columns) == SHIPMENT_COLUMNS
    assert len(df) == 3
    assert set(df["supplier"]) >= {"Aurobindo Pharma Limited", "Orgenics Ltd", "Trinity Biotech Plc"}
    assert (df["actual_delay_days"] >= 0).all()
    # First row delivered 5 days late.
    assert df.loc[df["supplier"] == "Aurobindo Pharma Limited", "actual_delay_days"].iloc[0] == 5.0
    # Peak season from December scheduled date.
    assert bool(df.loc[df["supplier"] == "Orgenics Ltd", "is_peak_season"].iloc[0]) is True


def test_transform_c2k_maps_required_columns(c2k_sample):
    df = transform_c2k(c2k_sample)
    assert list(df.columns) == SHIPMENT_COLUMNS
    assert len(df) == 3
    assert (df["actual_delay_days"] >= 0).all()
    # (2880-1440)/1440 = 1.0 day delay
    assert df.iloc[0]["actual_delay_days"] == pytest.approx(1.0)


def test_seed_database_round_trip(scms_sample):
    from week1.database import SessionLocal, init_db
    from week1.ingest_real_data import load_shipments, seed_database
    from week1.models import Shipment

    df = transform_scms(scms_sample)
    df["data_source"] = "usaid-scms"
    init_db()
    seed_database(df, replace=True)

    session = SessionLocal()
    try:
        assert session.query(Shipment).count() == len(df)
        assert session.query(Shipment).filter(Shipment.data_source == "usaid-scms").count() == len(df)
    finally:
        session.close()

    loaded = load_shipments(prefer_db=True)
    assert len(loaded) == len(df)
    assert set(loaded.columns) == set(SHIPMENT_COLUMNS)
