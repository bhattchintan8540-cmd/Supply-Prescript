"""
Week 2 - the prescriptive half of SupplyPrescript.

Turns a delay prediction into three concrete fulfillment options, plus
one blended allocation picked by an actual linear program (three
decision variables, two constraints - small, but it's a real LP, not
just three if/else branches dressed up as "optimization").

Channels
--------
AIR       - expedite everything, most expensive, fastest
SECONDARY - split the order to a backup supplier at a premium
DELAY     - do nothing differently, eat a holding-cost penalty for
            however many days the shipment is predicted to be late

These map directly onto Options A/B/C from the project brief (air
freight / secondary supplier / delay launch).
"""
from __future__ import annotations

from dataclasses import dataclass

import pulp

# --- business constants -------------------------------------------------
# Not pretending these are calibrated against real freight contracts -
# they're stand-ins tuned so the three options actually trade off against
# each other in the mock data (cheapest isn't always fastest, etc).
AIR_FREIGHT_SURCHARGE_PER_UNIT = 2.35
AIR_FREIGHT_HANDLING_FEE = 900.0
AIR_FREIGHT_RESULTING_DELAY_DAYS = 1.0

SECONDARY_SUPPLIER_PREMIUM_PCT = 0.10
SECONDARY_SUPPLIER_HANDLING_FEE = 450.0
SECONDARY_SUPPLIER_DELAY_FACTOR = 0.35  # fraction of the original delay that remains

HOLDING_COST_PER_UNIT_PER_DAY = 0.06


@dataclass
class ChannelQuote:
    label: str
    description: str
    cost_per_unit: float
    fixed_fee: float
    resulting_delay_days: float

    def total_cost(self, quantity: float) -> float:
        return self.cost_per_unit * quantity + self.fixed_fee


def _channel_quotes(unit_cost_usd: float, predicted_delay_days: float) -> dict[str, ChannelQuote]:
    return {
        "air_freight": ChannelQuote(
            label="Air Freight",
            description="Expedite the full order by air. Fastest option, highest per-unit cost.",
            cost_per_unit=unit_cost_usd + AIR_FREIGHT_SURCHARGE_PER_UNIT,
            fixed_fee=AIR_FREIGHT_HANDLING_FEE,
            resulting_delay_days=AIR_FREIGHT_RESULTING_DELAY_DAYS,
        ),
        "secondary_supplier": ChannelQuote(
            label="Secondary Supplier",
            description=f"Source from the backup supplier at a {SECONDARY_SUPPLIER_PREMIUM_PCT:.0%} premium.",
            cost_per_unit=unit_cost_usd * (1 + SECONDARY_SUPPLIER_PREMIUM_PCT),
            fixed_fee=SECONDARY_SUPPLIER_HANDLING_FEE,
            resulting_delay_days=round(predicted_delay_days * SECONDARY_SUPPLIER_DELAY_FACTOR, 1),
        ),
        "delay_launch": ChannelQuote(
            label="Delay Launch",
            description="Keep the original supplier, absorb the predicted delay, pay holding cost instead.",
            cost_per_unit=unit_cost_usd + HOLDING_COST_PER_UNIT_PER_DAY * predicted_delay_days,
            fixed_fee=0.0,
            resulting_delay_days=predicted_delay_days,
        ),
    }


def pure_options(
    unit_cost_usd: float,
    order_quantity: int,
    predicted_delay_days: float,
    budget_cap_usd: float,
) -> list[dict]:
    """The three 100%-allocated cards shown on the dashboard."""
    quotes = _channel_quotes(unit_cost_usd, predicted_delay_days)
    options = []
    for quote in quotes.values():
        cost = round(quote.total_cost(order_quantity), 2)
        options.append(
            {
                "label": quote.label,
                "description": quote.description,
                "cost_usd": cost,
                "resulting_delay_days": quote.resulting_delay_days,
                "within_budget": cost <= budget_cap_usd,
            }
        )
    return options


def solve_optimal_allocation(
    unit_cost_usd: float,
    order_quantity: int,
    predicted_delay_days: float,
    budget_cap_usd: float,
    max_acceptable_delay_days: float,
) -> dict:
    """Splits order_quantity across the three channels to minimize total
    cost while keeping the *quantity-weighted average delay* under
    max_acceptable_delay_days. Budget is enforced as a hard constraint
    first; if that makes the problem infeasible we relax it and flag
    the result instead of silently ignoring the cap.
    """
    quotes = _channel_quotes(unit_cost_usd, predicted_delay_days)
    channel_keys = list(quotes.keys())

    def _build_and_solve(enforce_budget: bool) -> tuple[pulp.LpProblem, dict[str, pulp.LpVariable]]:
        prob = pulp.LpProblem("supplyprescript_allocation", pulp.LpMinimize)
        x = {key: pulp.LpVariable(f"x_{key}", lowBound=0) for key in channel_keys}

        # Fixed handling fees are per-channel-used, not per-unit, which
        # would need extra binary "is this channel active" variables to
        # model exactly in the LP itself - overkill for a 3-channel demo.
        # The objective below optimizes on variable cost only; fixed fees
        # get added back on top once we know which channels came out
        # nonzero (see below the solve call).
        prob += pulp.lpSum(quotes[k].cost_per_unit * x[k] for k in channel_keys)

        prob += pulp.lpSum(x[k] for k in channel_keys) == order_quantity, "fulfil_full_order"
        prob += (
            pulp.lpSum(quotes[k].resulting_delay_days * x[k] for k in channel_keys)
            <= max_acceptable_delay_days * order_quantity
        ), "max_weighted_delay"

        if enforce_budget:
            prob += (
                pulp.lpSum(quotes[k].cost_per_unit * x[k] for k in channel_keys) <= budget_cap_usd
            ), "budget_cap"

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return prob, x

    prob, x = _build_and_solve(enforce_budget=True)
    budget_relaxed = False
    if pulp.LpStatus[prob.status] != "Optimal":
        # infeasible under the budget cap - solve again without it so the
        # manager still gets a recommendation, just clearly flagged as
        # over budget rather than returning nothing.
        prob, x = _build_and_solve(enforce_budget=False)
        budget_relaxed = True

    allocation = {k: round(v.value() or 0.0, 1) for k, v in x.items()}
    fixed_fees = sum(quotes[k].fixed_fee for k in channel_keys if allocation[k] > 0)
    variable_cost = sum(quotes[k].cost_per_unit * allocation[k] for k in channel_keys)
    total_cost = round(variable_cost + fixed_fees, 2)
    total_qty = sum(allocation.values()) or 1
    weighted_delay = round(
        sum(quotes[k].resulting_delay_days * allocation[k] for k in channel_keys) / total_qty, 1
    )

    return {
        "status": pulp.LpStatus[prob.status],
        "budget_relaxed": budget_relaxed,
        "allocation_units": allocation,
        "total_cost_usd": total_cost,
        "weighted_avg_delay_days": weighted_delay,
        "within_budget": total_cost <= budget_cap_usd,
    }
