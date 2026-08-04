from week2.solver import pure_options, solve_optimal_allocation


def test_pure_options_flags_over_budget_correctly():
    options = pure_options(unit_cost_usd=12.0, order_quantity=5000, predicted_delay_days=10.0, budget_cap_usd=70_000)
    labels = {o["label"] for o in options}
    assert labels == {"Air Freight", "Secondary Supplier", "Delay Launch"}
    for opt in options:
        expected_flag = opt["cost_usd"] <= 70_000
        assert opt["within_budget"] == expected_flag


def test_optimal_allocation_respects_budget_when_feasible():
    # generous budget and a loose delay ceiling - solver should find
    # something within budget rather than needing to relax it
    result = solve_optimal_allocation(
        unit_cost_usd=8.0,
        order_quantity=3000,
        predicted_delay_days=6.0,
        budget_cap_usd=100_000,
        max_acceptable_delay_days=10.0,
    )
    assert result["status"] == "Optimal"
    assert result["budget_relaxed"] is False
    assert result["total_cost_usd"] <= 100_000


def test_optimal_allocation_fulfills_full_order_quantity():
    result = solve_optimal_allocation(
        unit_cost_usd=15.0,
        order_quantity=2000,
        predicted_delay_days=9.0,
        budget_cap_usd=60_000,
        max_acceptable_delay_days=4.0,
    )
    total_units = sum(result["allocation_units"].values())
    # solver works in continuous units, tolerate float rounding
    assert abs(total_units - 2000) < 1.0


def test_optimal_allocation_meets_delay_ceiling():
    result = solve_optimal_allocation(
        unit_cost_usd=10.0,
        order_quantity=4000,
        predicted_delay_days=14.0,
        budget_cap_usd=250_000,
        max_acceptable_delay_days=3.0,
    )
    assert result["weighted_avg_delay_days"] <= 3.0 + 0.1  # small tolerance for rounding
