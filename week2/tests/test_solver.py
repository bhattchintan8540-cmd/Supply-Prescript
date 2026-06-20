from week2.solver import pure_options, solve_optimal_allocation


def test_pure_options_flags_over_budget_correctly():
    options = pure_options(
        unit_cost_usd=12.0,
        order_quantity=5000,
        predicted_delay_days=10.0,
        budget_cap_usd=70_000,
        predicted_delay_probability=0.8,
    )
    labels = {o["label"] for o in options}
    assert labels == {"Air Freight", "Secondary Supplier", "Delay Launch"}
    for opt in options:
        expected_flag = opt["cost_usd"] <= 70_000
        assert opt["within_budget"] == expected_flag


def test_delay_launch_cost_scales_with_probability():
    """Expected holding = P(delay) × rate × days — low probability must
    make 'do nothing' cheaper than the same magnitude at high probability."""
    low = pure_options(
        unit_cost_usd=10.0,
        order_quantity=1000,
        predicted_delay_days=10.0,
        budget_cap_usd=1_000_000,
        predicted_delay_probability=0.2,
    )
    high = pure_options(
        unit_cost_usd=10.0,
        order_quantity=1000,
        predicted_delay_days=10.0,
        budget_cap_usd=1_000_000,
        predicted_delay_probability=0.95,
    )
    low_delay = next(o for o in low if o["label"] == "Delay Launch")
    high_delay = next(o for o in high if o["label"] == "Delay Launch")
    assert low_delay["cost_usd"] < high_delay["cost_usd"]


def test_optimal_allocation_respects_budget_when_feasible():
    # generous budget and a loose delay ceiling - solver should find
    # something within budget rather than needing to relax it
    result = solve_optimal_allocation(
        unit_cost_usd=8.0,
        order_quantity=3000,
        predicted_delay_days=6.0,
        budget_cap_usd=100_000,
        max_acceptable_delay_days=10.0,
        predicted_delay_probability=0.7,
        partial_fulfillment_useful=True,
    )
    assert result["status"] == "Optimal"
    assert result["budget_relaxed"] is False
    assert result["total_cost_usd"] <= 100_000
    assert result["within_budget"] is True


def test_optimal_allocation_fulfills_full_order_quantity():
    result = solve_optimal_allocation(
        unit_cost_usd=15.0,
        order_quantity=2000,
        predicted_delay_days=9.0,
        budget_cap_usd=60_000,
        max_acceptable_delay_days=4.0,
        predicted_delay_probability=0.8,
        partial_fulfillment_useful=True,
    )
    total_units = sum(result["allocation_units"].values())
    # solver works in continuous units, tolerate float rounding
    assert abs(total_units - 2000) < 1.0


def test_weighted_average_mode_meets_delay_ceiling():
    result = solve_optimal_allocation(
        unit_cost_usd=10.0,
        order_quantity=4000,
        predicted_delay_days=14.0,
        budget_cap_usd=250_000,
        max_acceptable_delay_days=3.0,
        predicted_delay_probability=0.9,
        partial_fulfillment_useful=True,
    )
    assert result["delay_constraint_mode"] == "weighted_average"
    assert result["weighted_avg_delay_days"] <= 3.0 + 0.1


def test_makespan_mode_rejects_slow_channel_when_sla_is_tight():
    """If production waits for the last unit, a 9-day channel cannot be
    mixed in under a 5-day SLA — unlike weighted-average mode."""
    result = solve_optimal_allocation(
        unit_cost_usd=10.0,
        order_quantity=10_000,
        predicted_delay_days=9.0,
        budget_cap_usd=500_000,
        max_acceptable_delay_days=5.0,
        predicted_delay_probability=0.9,
        partial_fulfillment_useful=False,
    )
    assert result["delay_constraint_mode"] == "operational_makespan"
    assert result["operational_delay_days"] <= 5.0 + 0.1
    # delay_launch has 9d residual → must not be activated under makespan SLA
    assert result["allocation_units"]["delay_launch"] == 0.0


def test_fixed_fees_are_inside_budget_constraint():
    """Budget must include activation fees, not just variable cost.

    Choose a budget that covers variable air cost but not air + $900 fee
    when air is the only feasible channel under a tight makespan SLA.
    """
    unit_cost = 10.0
    qty = 1000
    # Air variable-ish expected cost ~ (10+2.35+0.06)*1000 = 12410, fee 900
    # Budget between variable-only and variable+fee forces either relaxation
    # or a different channel — either way within_budget must reflect fees.
    result = solve_optimal_allocation(
        unit_cost_usd=unit_cost,
        order_quantity=qty,
        predicted_delay_days=12.0,
        budget_cap_usd=12_500,  # covers air variable ~12.4k but not +900 fee
        max_acceptable_delay_days=2.0,
        predicted_delay_probability=1.0,
        partial_fulfillment_useful=False,
    )
    # Only air meets a 2-day makespan (air=1d). Fee pushes total over 12.5k.
    assert result["allocation_units"]["air_freight"] > 0
    assert result["total_cost_usd"] > 12_500
    assert result["within_budget"] is False
    assert result["budget_relaxed"] is True
    assert result["fixed_fees_usd"] >= 900.0
    assert result.get("infeasible") is False


def test_infeasible_makespan_returns_empty_allocation():
    """Impossible SLA must not invent a fake Optimal-looking plan."""
    result = solve_optimal_allocation(
        unit_cost_usd=10.0,
        order_quantity=1000,
        predicted_delay_days=12.0,
        budget_cap_usd=500_000,
        max_acceptable_delay_days=0.0,
        predicted_delay_probability=1.0,
        partial_fulfillment_useful=False,
    )
    assert result["status"] == "Infeasible"
    assert result["infeasible"] is True
    assert sum(result["allocation_units"].values()) == 0.0
    assert result["within_budget"] is False
    assert result["message"]


def test_pure_options_flag_sla_when_ceiling_provided():
    options = pure_options(
        unit_cost_usd=12.0,
        order_quantity=1000,
        predicted_delay_days=10.0,
        budget_cap_usd=200_000,
        predicted_delay_probability=0.9,
        max_acceptable_delay_days=5.0,
    )
    delay = next(o for o in options if o["label"] == "Delay Launch")
    air = next(o for o in options if o["label"] == "Air Freight")
    assert delay["within_sla"] is False
    assert air["within_sla"] is True
