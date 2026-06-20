"""Lightweight Week 5 checks that do not require a full model retrain."""

from week2.solver import pure_options, solve_optimal_allocation


def test_smoke_solver_path_returns_status():
    options = pure_options(
        unit_cost_usd=14.2,
        order_quantity=6000,
        predicted_delay_days=6.0,
        budget_cap_usd=100_000,
        predicted_delay_probability=0.8,
        max_acceptable_delay_days=5,
    )
    assert len(options) == 3
    assert all("within_sla" in o for o in options)

    blend = solve_optimal_allocation(
        unit_cost_usd=14.2,
        order_quantity=6000,
        predicted_delay_days=6.0,
        budget_cap_usd=100_000,
        max_acceptable_delay_days=5,
        predicted_delay_probability=0.8,
    )
    assert "status" in blend
    assert "infeasible" in blend
