import pandas as pd
import pytest

from week1.config import ROOT_DIR
from week1.delay_model import DelayModel

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"


@pytest.fixture(scope="module")
def trained_model():
    if not DATA_PATH.exists():
        pytest.skip(f"{DATA_PATH} missing - run scripts/generate_mock_data.py first")
    df = pd.read_csv(DATA_PATH)
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
