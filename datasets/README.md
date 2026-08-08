# Datasets used in SupplyPrescript

These are the **separate real open datasets** the project trains on.
They are not synthetic / mock rows.

| File | Source | Rows | What it is |
|---|---|---|---|
| `usaid_scms_shipments.csv` | USAID / PEPFAR SCMS | 10,307 | Health-commodity shipments mapped to the model schema (default training set) |
| `uci_c2k_shipments.csv` | UCI ML Repository Cargo 2000 | 3,942 | Real air-freight process instances mapped to the same schema |
| `combined_shipments.csv` | Both | 14,249 | Concatenation of the two extracts above |
| `raw_usaid_scms_delivery_history.csv` | USAID SCMS (raw) | 10,324 | Original columns before mapping |
| `raw_uci_c2k_freight_tracking.zip` | UCI C2K (raw) | 3,943 | Original Cargo 2000 extract from the UCI archive |

## Model schema columns (mapped CSVs)

`sku`, `supplier`, `origin_region`, `distance_km`, `historical_avg_lead_time_days`,
`order_quantity`, `unit_cost_usd`, `is_peak_season`, `actual_delay_days`

`actual_delay_days` is the training label (scheduled vs actual delivery, clipped at 0 for early arrivals).

## Upstream references

- USAID SCMS: [Supply Chain Shipment Pricing Dataset](https://data.usaid.gov/HIV-AIDS/Supply-Chain-Shipment-Pricing-Dataset/a3rc-nmf6)
- UCI Cargo 2000: [Cargo 2000 Freight Tracking and Tracing](https://archive.ics.uci.edu/dataset/382/cargo+2000+freight+tracking+and+tracing)

## Regenerate / re-download

```bash
python week1/ingest_real_data.py                 # USAID SCMS → data/ + DB
python week1/ingest_real_data.py --source uci-c2k
python week1/ingest_real_data.py --source both
```

Raw downloads are also cached under `data/raw/`.
