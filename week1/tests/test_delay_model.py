import pandas as pd
import pytest

from week1.config import ROOT_DIR
from week1.delay_model import DelayModel
from week1.ingest_real_data import load_shipments


@pytest.fixture(scope="module")
def trained_model():
    try:
        df = load_shipments(prefer_db=True)
    except FileNotFoundError:
        pytest.skip("No shipment data — run: python week1/ingest_real_data.py")
    if len(df) < 50:
        pytest.skip("Shipment dataset too small to train")
    model = DelayModel()
    model.fit(df, verbose=False)
    return model, df


def test_predict_one_returns_sane_ranges(trained_model):
    model, df = trained_model
    sample = df.iloc[0].drop("actual_delay_days").to_dict()
    days, prob = model.predict_one(sample)
    assert days >= 0
    assert 0.0 <= prob <= 1.0


def test_save_and_load_round_trips_predictions(tmp_path, trained_model):
    model, df = trained_model
    sample = df.iloc[5].drop("actual_delay_days").to_dict()
    before = model.predict_one(sample)

    path = tmp_path / "model.joblib"
    model.save(path)
    reloaded = DelayModel.load(path)
    after = reloaded.predict_one(sample)

    assert before == pytest.approx(after)


def test_unseen_category_does_not_crash(trained_model):
    """A supplier that never appeared in training still has to produce a
    prediction, not a KeyError, once the business onboards someone new."""
    model, df = trained_model
    sample = df.iloc[0].drop("actual_delay_days").to_dict()
    sample["supplier"] = "Brand New Vendor Co"
    days, prob = model.predict_one(sample)
    assert days >= 0
    assert 0.0 <= prob <= 1.0


def test_feature_importance_returns_ranked_rows(trained_model):
    model, _ = trained_model
    rows = model.feature_importance(top_n=5)
    assert len(rows) == 5
    assert all("feature" in row and "importance" in row for row in rows)
    scores = [row["importance"] for row in rows]
    assert scores == sorted(scores, reverse=True)
