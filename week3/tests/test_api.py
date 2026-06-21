import pandas as pd
import pytest
from fastapi.testclient import TestClient

from week1.config import ROOT_DIR, MODEL_PATH
from week1.delay_model import DelayModel

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"


@pytest.fixture(scope="session", autouse=True)
def _ensure_model_artifact():
    """The API lazy-loads the model from disk, so tests need a real
    artifact sitting at MODEL_PATH - train a throwaway one if it's
    missing rather than requiring a manual step before `pytest`."""
    if not MODEL_PATH.exists():
        if not DATA_PATH.exists():
            pytest.skip(f"{DATA_PATH} missing - run week1/generate_mock_data.py first")
        df = pd.read_csv(DATA_PATH)
        model = DelayModel()
        model.fit(df, verbose=False)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)


@pytest.fixture(scope="module")
def client():
    from week3.main import app  # imported here so the DATABASE_URL env override in conftest wins

    with TestClient(app) as c:
        yield c


SAMPLE_SHIPMENT = {
    "sku": "MICROCHIP-A2",
    "supplier": "NovaChip Manufacturing",
    "origin_region": "Asia Pacific",
    "distance_km": 8800,
    "historical_avg_lead_time_days": 16,
    "order_quantity": 6000,
    "unit_cost_usd": 14.2,
    "is_peak_season": False,
}


def test_health(client):
    assert client.get("/health").status_code == 200


def test_model_info_endpoint(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert "model_path" in body
    # metrics.json may or may not exist in CI; either way the payload is valid
    assert "mae_days" in body
    assert body.get("data_is_synthetic") is True
    assert isinstance(body.get("top_features", []), list)
    assert "segment_auc" in body


def test_predict_returns_a_prediction(client):
    """Software correctness: probability is in [0, 1]. This does not
    prove the probability is calibrated or analytically accurate."""
    resp = client.post("/predict", json=SAMPLE_SHIPMENT)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_delay_days"] >= 0
    assert 0 <= body["predicted_delay_probability"] <= 1


def test_prescribe_returns_four_options(client):
    resp = client.post("/prescribe", json={"shipment": SAMPLE_SHIPMENT})
    assert resp.status_code == 200
    body = resp.json()
    labels = {o["label"] for o in body["options"]}
    assert labels == {"Air Freight", "Secondary Supplier", "Delay Launch", "Optimizer Recommended Split"}
    assert body["no_action_cost_usd"] is not None
    assert body["delay_constraint_mode"] in {"operational_makespan", "weighted_average"}


def test_full_decision_lifecycle_cost_accuracy_and_roi(client):
    prescribe_resp = client.post("/prescribe", json={"shipment": SAMPLE_SHIPMENT})
    body = prescribe_resp.json()
    # Choose an intervention (not Delay Launch) so ROI has a counterfactual gap.
    chosen = next(o for o in body["options"] if o["label"] == "Air Freight")

    create_resp = client.post(
        "/decisions",
        json={
            "shipment_sku": SAMPLE_SHIPMENT["sku"],
            "predicted_delay_days": body["prediction"]["predicted_delay_days"],
            "predicted_delay_probability": body["prediction"]["predicted_delay_probability"],
            "options": body["options"],
            "chosen_option_label": chosen["label"],
            "budget_cap_usd": body["budget_cap_usd"],
            "shipment_features": SAMPLE_SHIPMENT,
            "no_action_cost_usd": body["no_action_cost_usd"],
        },
    )
    assert create_resp.status_code == 201
    decision = create_resp.json()
    assert decision["is_resolved"] is False
    assert decision["no_action_cost_usd"] == body["no_action_cost_usd"]

    # Actual cost slightly under the no-action baseline → positive avoided loss.
    actual_cost = min(decision["predicted_cost_usd"] * 1.05, body["no_action_cost_usd"] * 0.9)
    outcome_resp = client.patch(
        f"/decisions/{decision['id']}/outcome",
        json={"actual_cost_usd": actual_cost, "actual_delay_days": 1.0},
    )
    assert outcome_resp.status_code == 200
    assert outcome_resp.json()["is_resolved"] is True

    accuracy = client.get("/decisions/cost-accuracy").json()
    assert accuracy["resolved_decisions"] >= 1
    assert accuracy["avg_cost_error_pct"] is not None

    roi = client.get("/decisions/roi").json()
    assert roi["decisions_with_counterfactual"] >= 1
    assert roi["avg_avoided_loss_usd"] is not None
    assert roi["avg_roi_pct"] is not None


def test_rejects_unknown_option_label(client):
    resp = client.post(
        "/decisions",
        json={
            "shipment_sku": "X",
            "predicted_delay_days": 3.0,
            "predicted_delay_probability": 0.5,
            "options": [{"label": "Air Freight", "description": "x", "cost_usd": 100, "resulting_delay_days": 1, "within_budget": True}],
            "chosen_option_label": "Not A Real Option",
            "budget_cap_usd": 1000,
        },
    )
    assert resp.status_code == 422


def test_dashboard_is_served(client):
    resp = client.get("/ui/", follow_redirects=True)
    assert resp.status_code == 200
    assert "SupplyPrescript" in resp.text
