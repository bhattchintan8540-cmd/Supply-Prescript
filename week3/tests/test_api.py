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
    assert "fit_quality" in body


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
    assert "app.js" in resp.text


def test_root_and_ui_without_slash_open_in_browser(client):
    """Uvicorn prints http://127.0.0.1:8000 — that URL must reach the UI."""
    root = client.get("/", follow_redirects=False)
    assert root.status_code in {301, 302, 303, 307, 308}
    assert root.headers["location"].rstrip("/").endswith("/ui") or root.headers["location"].endswith("/ui/")

    noslash = client.get("/ui", follow_redirects=False)
    assert noslash.status_code in {301, 302, 303, 307, 308}
    assert "/ui/" in noslash.headers["location"]

    page = client.get("/", follow_redirects=True)
    assert page.status_code == 200
    assert "SupplyPrescript" in page.text


def test_phase2_recommend_endpoint(client):
    resp = client.get("/phase2/recommend")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sku"]
    assert body["grid_cells"] == 9
    assert len(body["grid"]) == 9
    assert body["winner_label"] in {
        "Air Freight",
        "Secondary Supplier",
        "Delay Launch",
        None,
    }
    assert 1 <= body["same_winner_cells"] <= body["grid_cells"]


def test_phase2_recommend_query_overrides(client):
    demo = client.get("/phase2/recommend")
    assert demo.status_code == 200
    demo_body = demo.json()
    assert demo_body["budget_cap_usd"] == 100_000
    assert demo_body["max_acceptable_delay_days"] == 5
    demo_cell = _grid_cell(demo_body, 100_000, 5)
    assert demo_body["winner_label"] == demo_cell["winner_label"]
    assert demo_body["winner_cost_usd"] == demo_cell["winner_cost_usd"]

    overridden = client.get("/phase2/recommend", params={"budget": 120_000, "max_delay": 8})
    assert overridden.status_code == 200
    body = overridden.json()
    assert body["budget_cap_usd"] == 120_000
    assert body["max_acceptable_delay_days"] == 8
    assert body["grid_cells"] == 9
    cell = _grid_cell(body, 120_000, 8)
    assert body["winner_label"] == cell["winner_label"]
    assert body["winner_cost_usd"] == cell["winner_cost_usd"]
    assert body["milp_feasible"] == cell["milp_feasible"]

    budget_only = client.get("/phase2/recommend", params={"budget": 80_000})
    assert budget_only.status_code == 200
    budget_body = budget_only.json()
    assert budget_body["budget_cap_usd"] == 80_000
    assert budget_body["max_acceptable_delay_days"] == 5
    tight = _grid_cell(budget_body, 80_000, 5)
    assert budget_body["winner_label"] == tight["winner_label"]
    assert budget_body["winner_cost_usd"] == tight["winner_cost_usd"]

    off_grid = client.get("/phase2/recommend", params={"budget": 90_000, "max_delay": 4})
    assert off_grid.status_code == 200
    off = off_grid.json()
    assert off["budget_cap_usd"] == 90_000
    assert off["max_acceptable_delay_days"] == 4
    assert off["grid_cells"] == 9
    assert off["winner_label"] in {
        "Air Freight",
        "Secondary Supplier",
        "Delay Launch",
        None,
    }


def _grid_cell(body: dict, budget: float, delay: float) -> dict:
    return next(
        cell
        for cell in body["grid"]
        if cell["budget_cap_usd"] == budget and cell["max_acceptable_delay_days"] == delay
    )


def test_phase2_grid_endpoint(client):
    resp = client.get("/phase2/grid")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sku"] == "MICROCHIP-A2"
    assert body["grid_cells"] == 9
    assert len(body["grid"]) == 9
    assert {cell["budget_cap_usd"] for cell in body["grid"]} == {80_000.0, 100_000.0, 120_000.0}
    assert {cell["max_acceptable_delay_days"] for cell in body["grid"]} == {3.0, 5.0, 8.0}
    for cell in body["grid"]:
        assert cell["winner_label"] in {
            "Air Freight",
            "Secondary Supplier",
            "Delay Launch",
            None,
        }
        assert isinstance(cell["milp_feasible"], bool)


def test_phase2_draft_decision_is_preview_only(client):
    resp = client.post("/phase2/draft-decision", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["persisted"] is False
    assert body["shipment_sku"] == "MICROCHIP-A2"
    assert body["budget_cap_usd"] == 100_000
    assert body["max_acceptable_delay_days"] == 5
    assert client.get("/decisions").json() == []

    overridden = client.post(
        "/phase2/draft-decision",
        json={"budget": 120_000, "max_delay": 8},
    )
    assert overridden.status_code == 200
    preview = overridden.json()
    assert preview["budget_cap_usd"] == 120_000
    assert preview["max_acceptable_delay_days"] == 8
    assert preview["persisted"] is False
    assert client.get("/decisions").json() == []
