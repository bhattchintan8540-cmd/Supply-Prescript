"""
Week 2 - the prescriptive half of SupplyPrescript.

Turns a delay prediction into three concrete fulfillment options, plus
one blended allocation picked by a mixed-integer program (quantity
variables + channel-activation binaries).

Why probability enters the objective
------------------------------------
The Week 1 classifier estimates P(significant delay). The regressor
estimates delay magnitude. Expected financial impact of "do nothing" is:

    expected holding cost = P(delay) × holding_rate × predicted_days

Interventions that change the outcome (air freight) treat their residual
delay as essentially certain once chosen. Secondary supplier residual
risk is still scaled by P(delay) because that channel is a scenario-based
proxy, not a qualified supplier selection.

Why fixed fees are inside the model
-----------------------------------
Air freight and secondary supplier carry fixed handling fees. Those used
to be added after a continuous LP, which meant the budget constraint
could accept a solution that later failed once fees were applied. The
MILP activates a binary per channel and includes fee_k * y_k in both the
objective and the budget constraint.

Why the delay constraint is not only a weighted average
-------------------------------------------------------
A quantity-weighted average delay of five days can hide a 9-day lag on
half the order. If production cannot start until every unit arrives,
the operational delay is the makespan (max delay across used channels).
That is the default. Weighted-average mode remains available when
partial fulfillment creates usable business value
(partial_fulfillment_useful=True).

Channels
--------
AIR       - expedite everything, most expensive, fastest
SECONDARY - scenario-based backup-supplier option at a premium
            (not a real supplier-selection engine — see docs/business/)
DELAY     - do nothing differently; absorb expected holding cost
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
    # Expected per-unit cost contribution used by the optimizer
    # (includes probability-weighted holding where applicable).
    expected_cost_per_unit: float

    def total_cost(self, quantity: float) -> float:
        """Deterministic quote card total (variable + fixed)."""
        return self.cost_per_unit * quantity + self.fixed_fee

    def expected_total_cost(self, quantity: float) -> float:
        """Expected total used for decision economics / ROI counterfactual."""
        return self.expected_cost_per_unit * quantity + self.fixed_fee


def _channel_quotes(
    unit_cost_usd: float,
    predicted_delay_days: float,
    predicted_delay_probability: float,
) -> dict[str, ChannelQuote]:
    """Build channel quotes using expected financial impact.

    Delay Launch expected holding = P(delay) × rate × magnitude.
    Air freight residual delay is treated as certain once activated.
    Secondary residual risk remains probability-scaled (scenario option).
    """
    p = max(0.0, min(1.0, float(predicted_delay_probability)))
    d = max(0.0, float(predicted_delay_days))

    # Delay launch: original unit cost + expected holding loss.
    delay_expected_holding = p * HOLDING_COST_PER_UNIT_PER_DAY * d
    # Display cost also uses expected holding so the card matches the LP.
    delay_cost_per_unit = unit_cost_usd + delay_expected_holding

    # Air: residual 1-day delay is operationally nearly certain once chosen.
    air_cost_per_unit = unit_cost_usd + AIR_FREIGHT_SURCHARGE_PER_UNIT
    air_expected_per_unit = air_cost_per_unit + HOLDING_COST_PER_UNIT_PER_DAY * AIR_FREIGHT_RESULTING_DELAY_DAYS

    # Secondary: premium + expected residual holding (still uncertain).
    secondary_residual_days = round(d * SECONDARY_SUPPLIER_DELAY_FACTOR, 1)
    secondary_cost_per_unit = unit_cost_usd * (1 + SECONDARY_SUPPLIER_PREMIUM_PCT)
    secondary_expected_per_unit = (
        secondary_cost_per_unit + p * HOLDING_COST_PER_UNIT_PER_DAY * secondary_residual_days
    )

    return {
        "air_freight": ChannelQuote(
            label="Air Freight",
            description=(
                "Expedite the full order by air. Fastest option, highest per-unit cost. "
                "Residual delay treated as certain once activated."
            ),
            cost_per_unit=air_expected_per_unit,
            fixed_fee=AIR_FREIGHT_HANDLING_FEE,
            resulting_delay_days=AIR_FREIGHT_RESULTING_DELAY_DAYS,
            expected_cost_per_unit=air_expected_per_unit,
        ),
        "secondary_supplier": ChannelQuote(
            label="Secondary Supplier",
            description=(
                f"Scenario-based backup option at a {SECONDARY_SUPPLIER_PREMIUM_PCT:.0%} premium "
                f"(not a qualified supplier-selection engine). Residual risk scaled by "
                f"P(delay)={p:.0%}."
            ),
            cost_per_unit=secondary_expected_per_unit,
            fixed_fee=SECONDARY_SUPPLIER_HANDLING_FEE,
            resulting_delay_days=secondary_residual_days,
            expected_cost_per_unit=secondary_expected_per_unit,
        ),
        "delay_launch": ChannelQuote(
            label="Delay Launch",
            description=(
                f"Keep the original supplier. Expected holding cost = "
                f"P(delay)×${HOLDING_COST_PER_UNIT_PER_DAY}/unit/day×{d:.1f}d "
                f"(P={p:.0%})."
            ),
            cost_per_unit=delay_cost_per_unit,
            fixed_fee=0.0,
            resulting_delay_days=d,
            expected_cost_per_unit=delay_cost_per_unit,
        ),
    }


def pure_options(
    unit_cost_usd: float,
    order_quantity: int,
    predicted_delay_days: float,
    budget_cap_usd: float,
    predicted_delay_probability: float = 1.0,
) -> list[dict]:
    """The three 100%-allocated cards shown on the dashboard.

    Costs are expected costs (probability enters Delay Launch / secondary).
    """
    quotes = _channel_quotes(unit_cost_usd, predicted_delay_days, predicted_delay_probability)
    options = []
    for quote in quotes.values():
        cost = round(quote.expected_total_cost(order_quantity), 2)
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
    predicted_delay_probability: float = 1.0,
    partial_fulfillment_useful: bool = False,
    min_on_time_fraction: float = 0.0,
) -> dict:
    """MILP: split order_quantity across channels to minimize expected
    total cost (variable + fixed activation fees) subject to:

    - fulfill the full order
    - operational delay constraint (makespan by default; weighted average
      only when partial_fulfillment_useful=True)
    - optional minimum on-time fill fraction
    - budget including fixed fees (relaxed + flagged if infeasible)

    Binary y_k indicates whether channel k is activated so fixed fees
    enter the objective and budget directly.
    """
    quotes = _channel_quotes(unit_cost_usd, predicted_delay_days, predicted_delay_probability)
    channel_keys = list(quotes.keys())
    Q = float(order_quantity)
    D_max = float(max_acceptable_delay_days)
    # Big-M for delay makespan linking: delay_k <= D_max + slack room.
    delay_values = [quotes[k].resulting_delay_days for k in channel_keys]
    big_m_delay = max(delay_values + [D_max]) + 1.0

    def _build_and_solve(enforce_budget: bool) -> tuple[pulp.LpProblem, dict, dict, pulp.LpVariable | None]:
        prob = pulp.LpProblem("supplyprescript_allocation", pulp.LpMinimize)
        x = {
            key: pulp.LpVariable(f"x_{key}", lowBound=0, upBound=Q, cat="Continuous")
            for key in channel_keys
        }
        y = {
            key: pulp.LpVariable(f"y_{key}", cat="Binary")
            for key in channel_keys
        }

        # Expected variable cost + fixed activation fees in the objective.
        prob += pulp.lpSum(
            quotes[k].expected_cost_per_unit * x[k] + quotes[k].fixed_fee * y[k]
            for k in channel_keys
        )

        prob += pulp.lpSum(x[k] for k in channel_keys) == Q, "fulfill_full_order"

        # Link quantity to activation: cannot ship on a channel without paying its fee.
        for k in channel_keys:
            prob += x[k] <= Q * y[k], f"activate_{k}"

        operational_delay = None
        if partial_fulfillment_useful:
            # Partial lots create usable value → quantity-weighted average is meaningful.
            prob += (
                pulp.lpSum(quotes[k].resulting_delay_days * x[k] for k in channel_keys)
                <= D_max * Q
            ), "max_weighted_delay"
        else:
            # Production waits for the last unit → constrain makespan.
            operational_delay = pulp.LpVariable("operational_delay_days", lowBound=0)
            for k in channel_keys:
                # If channel k is used, operational delay >= that channel's delay.
                prob += (
                    operational_delay >= quotes[k].resulting_delay_days - big_m_delay * (1 - y[k])
                ), f"makespan_link_{k}"
            prob += operational_delay <= D_max, "max_operational_delay"

        # Service-level style rule: enough quantity on channels within SLA.
        if min_on_time_fraction > 0:
            on_time_keys = [k for k in channel_keys if quotes[k].resulting_delay_days <= D_max]
            if on_time_keys:
                prob += (
                    pulp.lpSum(x[k] for k in on_time_keys) >= min_on_time_fraction * Q
                ), "min_on_time_fill"
            else:
                # No channel meets the SLA alone — force infeasibility under this rule
                # so the caller can relax budget / inspect status.
                prob += 0 >= 1, "min_on_time_impossible"

        if enforce_budget:
            prob += (
                pulp.lpSum(
                    quotes[k].expected_cost_per_unit * x[k] + quotes[k].fixed_fee * y[k]
                    for k in channel_keys
                )
                <= budget_cap_usd
            ), "budget_cap_including_fees"

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return prob, x, y, operational_delay

    prob, x, y, operational_delay_var = _build_and_solve(enforce_budget=True)
    budget_relaxed = False
    if pulp.LpStatus[prob.status] != "Optimal":
        # infeasible under the budget cap - solve again without it so the
        # manager still gets a recommendation, just clearly flagged as
        # over budget rather than returning nothing.
        prob, x, y, operational_delay_var = _build_and_solve(enforce_budget=False)
        budget_relaxed = True

    allocation = {k: round(v.value() or 0.0, 1) for k, v in x.items()}
    activated = {k: bool(round(y[k].value() or 0.0)) for k in channel_keys}

    fixed_fees = sum(quotes[k].fixed_fee for k in channel_keys if activated[k])
    variable_cost = sum(quotes[k].expected_cost_per_unit * allocation[k] for k in channel_keys)
    total_cost = round(variable_cost + fixed_fees, 2)
    total_qty = sum(allocation.values()) or 1.0
    weighted_delay = round(
        sum(quotes[k].resulting_delay_days * allocation[k] for k in channel_keys) / total_qty, 1
    )

    if partial_fulfillment_useful:
        operational_delay = weighted_delay
        delay_constraint_mode = "weighted_average"
    else:
        used_delays = [quotes[k].resulting_delay_days for k in channel_keys if allocation[k] > 0]
        operational_delay = round(max(used_delays) if used_delays else 0.0, 1)
        if operational_delay_var is not None and operational_delay_var.value() is not None:
            operational_delay = round(float(operational_delay_var.value()), 1)
        delay_constraint_mode = "operational_makespan"

    # Resulting delay reported on the optimizer card: the operationally
    # relevant figure for the active constraint mode.
    reported_delay = operational_delay if not partial_fulfillment_useful else weighted_delay

    return {
        "status": pulp.LpStatus[prob.status],
        "budget_relaxed": budget_relaxed,
        "allocation_units": allocation,
        "channels_activated": activated,
        "total_cost_usd": total_cost,
        "fixed_fees_usd": round(fixed_fees, 2),
        "weighted_avg_delay_days": weighted_delay,
        "operational_delay_days": operational_delay,
        "delay_constraint_mode": delay_constraint_mode,
        "resulting_delay_days": reported_delay,
        "within_budget": total_cost <= budget_cap_usd,
        "predicted_delay_probability_used": round(float(predicted_delay_probability), 3),
    }
