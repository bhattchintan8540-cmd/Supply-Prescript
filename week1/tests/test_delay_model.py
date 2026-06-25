import pandas as pd
import pytest

from week1.config import ROOT_DIR
from week1.delay_model import DelayModel
from week1.generate_mock_data import build

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"


@pytest.fixture(scope="module")
def trained_model():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = build(n_rows=800)
    model = DelayModel()
    metrics = model.fit(df, verbose=False)
    return model, df, metrics


def test_predict_one_returns_sane_ranges(trained_model):
    """Software correctness check — range validity, not analytical accuracy."""
    model, df, _ = trained_model
    sample = df.iloc[0].drop(labels=["actual_delay_days"], errors="ignore").to_dict()
    sample.pop("shipment_date", None)
    days, prob = model.predict_one(sample)
    assert days >= 0
    assert 0.0 <= prob <= 1.0


def test_save_and_load_round_trips_predictions(tmp_path, trained_model):
    model, df, _ = trained_model
    sample = df.iloc[5].drop(labels=["actual_delay_days"], errors="ignore").to_dict()
    sample.pop("shipment_date", None)
    before = model.predict_one(sample)

    path = tmp_path / "model.joblib"
    model.save(path)
    reloaded = DelayModel.load(path)
    after = reloaded.predict_one(sample)

    assert before == pytest.approx(after)


def test_unseen_category_does_not_crash(trained_model):
    """A supplier that never appeared in training still has to produce a
    prediction, not a KeyError, once the business onboards someone new."""
    model, df, _ = trained_model
    sample = df.iloc[0].drop(labels=["actual_delay_days"], errors="ignore").to_dict()
    sample.pop("shipment_date", None)
    sample["supplier"] = "Brand New Vendor Co"
    days, prob = model.predict_one(sample)
    assert days >= 0
    assert 0.0 <= prob <= 1.0


def test_feature_importance_returns_ranked_rows(trained_model):
    model, _, _ = trained_model
    rows = model.feature_importance(top_n=5)
    assert len(rows) == 5
    assert all("feature" in row and "importance" in row for row in rows)
    scores = [row["importance"] for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_fit_uses_temporal_split_when_dates_present(trained_model):
    _, df, metrics = trained_model
    if "shipment_date" not in df.columns:
        pytest.skip("no shipment_date column")
    assert metrics["validation_strategy"] == "temporal_60_20_20"
    assert metrics["n_train"] > 0 and metrics["n_test"] > 0
    assert metrics["n_val"] > 0
    assert metrics["data_is_synthetic"] is True
    assert metrics["validation_used_for_tuning"] is True
    assert 0.2 <= metrics["decision_threshold"] <= 0.8


def test_fit_reports_baseline_and_classifier_diagnostics(trained_model):
    _, _, metrics = trained_model
    assert "baseline_mae_days" in metrics
    assert "precision" in metrics and "recall" in metrics and "f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["baseline_mae_days"] >= 0
    assert metrics["fit_quality"] in {"balanced", "overfit", "underfit", "unvalidated"}
    assert metrics["mae_train"] is not None
    assert metrics["mae_val"] is not None
    cap = metrics["n_estimators_cap"]
    used = metrics["n_estimators_regressor"]
    if cap is not None and used is not None:
        assert used <= cap
    assert 2 <= int(metrics["max_depth"]) <= 5


def test_capacity_helpers_label_overfit_and_underfit():
    from week1.delay_model import adjust_booster_params, diagnose_capacity, _BASE_BOOSTER

    assert diagnose_capacity(0.4, 2.0, 1.5, 1.4) == "overfit"
    assert diagnose_capacity(1.5, 1.6, 1.5, 1.4) == "underfit"
    assert diagnose_capacity(1.0, 1.1, 1.8, 1.7) == "balanced"
    over = adjust_booster_params(_BASE_BOOSTER, "overfit")
    under = adjust_booster_params(_BASE_BOOSTER, "underfit")
    assert over["max_depth"] == _BASE_BOOSTER["max_depth"] - 1
    assert over["reg_lambda"] > _BASE_BOOSTER["reg_lambda"]
    assert under["max_depth"] == _BASE_BOOSTER["max_depth"] + 1
    assert adjust_booster_params(_BASE_BOOSTER, "balanced") is None


def test_trained_model_includes_month_cycle_features(trained_model):
    model, _, _ = trained_model
    assert "month_sin" in model.feature_columns
    assert "month_cos" in model.feature_columns


def test_save_and_load_round_trips_calibrator(tmp_path, trained_model):
    model, df, metrics = trained_model
    sample = df.iloc[5].drop(labels=["actual_delay_days"], errors="ignore").to_dict()
    sample.pop("shipment_date", None)
    sample["is_peak_season"] = "True"
    before = model.predict_one(sample)

    path = tmp_path / "model.joblib"
    model.save(path)
    reloaded = DelayModel.load(path)
    after = reloaded.predict_one(sample)

    assert before == pytest.approx(after)
    assert reloaded.decision_threshold == pytest.approx(model.decision_threshold)
    assert metrics["validation_used_for_tuning"] is True
