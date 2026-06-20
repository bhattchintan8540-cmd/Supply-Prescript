# Production Data Requirements

The prototype uses a compact feature set that is enough to demonstrate
the decision loop on synthetic data. A production delay model and
prescriptive engine would need richer inputs. You do **not** need to
implement all of these for the portfolio — but you should be able to
discuss them.

## Predictive model — additional features to discuss

| Domain | Examples | Why it matters |
|---|---|---|
| Dates | PO date, requested / promised delivery | Lead-time slip vs promise, not just absolute delay |
| Carrier / mode / lane | Carrier, mode, shipping lane, port | Congestion and mode-specific variance |
| Supplier performance | OTIF history, capacity signals | Beyond a static reliability prior |
| Inventory | On-hand, safety stock, open POs | Shortage risk depends on buffer |
| Demand / production | Batch size, need-by date, criticality | Links delay to plant impact |
| Trade | Customs, port dwell | Especially for Asia→US/EU lanes |
| Alternates | Qualified alts, capacity, MOQ, contract | Turns "secondary supplier" into selection |

## Prescriptive engine — operational requirements

| Requirement | Effect on math |
|---|---|
| Partial fulfillment useful? | Makespan vs weighted-average delay |
| Minimum quantity by date | Hard fill constraints / on-time fraction |
| Shortage cost / criticality | Objective weights beyond holding |
| Inventory already available | Net requirement `< Q` |
| Alternate capacity / MOQ | Bounds on secondary `x_k` |
| Fixed activation costs | Binary variables (already in prototype MILP) |

## Secondary supplier — prototype vs production

**Prototype:** scenario option with +10% premium and reduced residual delay.

**Production:** which supplier, SKU qualification, capacity, inventory,
lead time, contract, MOQ — i.e. a supplier-selection problem, not a
single assumed channel.
