"""The Phase 2 draft stays in memory and matches the midpoint recommendation."""

from week6.phase2_midpoint import DEMO_BUDGET_USD, DEMO_MAX_DELAY_DAYS, SAMPLE
from week7.draft import draft_decision
from week7.recommend import recommend_midpoint


def test_draft_matches_recommendation_and_is_not_persisted():
    result = recommend_midpoint()
    draft = draft_decision(result)

    assert draft["shipment_sku"] == result["sku"] == SAMPLE["sku"]
    assert draft["chosen_option_label"] == result["winner_label"]
    assert draft["predicted_cost_usd"] == result["winner_cost_usd"]
    assert draft["budget_cap_usd"] == DEMO_BUDGET_USD
    assert draft["max_acceptable_delay_days"] == DEMO_MAX_DELAY_DAYS
    assert draft["shipment_features"]["sku"] == SAMPLE["sku"]
    assert {opt["label"] for opt in draft["options"]} == {
        "Air Freight",
        "Secondary Supplier",
        "Delay Launch",
    }
    assert draft["no_action_cost_usd"] is not None
    assert draft["actual_cost_usd"] is None
    assert draft["actual_delay_days"] is None
    assert draft["persisted"] is False
    assert "id" not in draft
