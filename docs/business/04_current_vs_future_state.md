# Current-State vs Future-State Process

## Current state (typical manual process)

1. Planner notices a shipment "feels late" from email / tribal knowledge.
2. Expedite is requested ad hoc (often air freight by default).
3. Budget impact is checked after the fact.
4. Outcome (did the line still wait?) is rarely linked back to the choice.
5. No systematic comparison to "what if we had done nothing?"

## Future state (this prototype)

1. Shipment features → delay **probability** + **magnitude**.
2. Expected financial impact of Delay Launch = P(delay) × holding × days.
3. Planner sees three pure options plus a MILP mix that:
   - includes fixed activation fees in the budget
   - respects operational makespan (or weighted average when partial fill helps)
4. Human selects an option → write-back to `decisions`.
5. Later, actual cost/delay are logged.
6. Measurement:
   - **ROI** vs stored no-action cost
   - **Cost accuracy** separately
7. Drift trigger + retrain on shipments ∪ eligible outcomes.

## Gap this still leaves for production

Alternate-supplier qualification, inventory on hand, PO/promise dates,
carrier/lane data, and plant batch rules are discussed in
`07_production_data_requirements.md` — intentionally out of prototype scope.
