"""
Adapters that map real open datasets onto SupplyPrescript's shipment schema.

Sources (downloaded at ingest time, not fabricated):
  - usaids-scms  — USAID / PEPFAR Supply Chain Management System delivery
                   history (~10k health-commodity shipments). Public open data.
  - uci-c2k      — UCI Machine Learning Repository “Cargo 2000 Freight Tracking
                   and Tracing” (~3.9k real air-freight process instances).

Both adapters emit the same columns the delay model already trains on, so the
rest of the pipeline (features → XGBoost → solver → API) stays unchanged.
"""
from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public download URLs (primary + mirrors). Tried in order.
# ---------------------------------------------------------------------------
SCMS_URLS = [
    # USAID Development Data Library (Socrata) — authoritative when reachable.
    "https://data.usaid.gov/resource/a3rc-nmf6.csv?$limit=50000",
    # Stable public mirror of the same SCMS delivery-history extract.
    "https://raw.githubusercontent.com/shashank-sundi/HEALTH-COMMODITY-SHIPMENT-PRICE-PREDICTION/master/SCMS_Delivery_History_Dataset.csv",
]

C2K_URLS = [
    "https://archive.ics.uci.edu/static/public/382/cargo+2000+freight+tracking+and+tracing.zip",
]

SHIPMENT_COLUMNS = [
    "sku",
    "supplier",
    "origin_region",
    "distance_km",
    "historical_avg_lead_time_days",
    "order_quantity",
    "unit_cost_usd",
    "is_peak_season",
    "actual_delay_days",
]

# Rough great-circle style corridor distances (km) by origin → destination region.
CORRIDOR_KM = {
    ("Asia Pacific", "Africa"): 8200,
    ("Asia Pacific", "Asia Pacific"): 3800,
    ("Asia Pacific", "Europe"): 7800,
    ("Asia Pacific", "Latin America"): 16000,
    ("Asia Pacific", "North America"): 12000,
    ("Asia Pacific", "Middle East"): 4500,
    ("Europe", "Africa"): 6200,
    ("Europe", "Asia Pacific"): 7800,
    ("Europe", "Europe"): 1400,
    ("Europe", "Latin America"): 9500,
    ("Europe", "North America"): 6500,
    ("Europe", "Middle East"): 3800,
    ("North America", "Africa"): 11000,
    ("North America", "Asia Pacific"): 12000,
    ("North America", "Europe"): 6500,
    ("North America", "Latin America"): 4200,
    ("North America", "North America"): 2200,
    ("North America", "Middle East"): 10500,
    ("Middle East", "Africa"): 4500,
    ("Middle East", "Asia Pacific"): 5500,
    ("Middle East", "Europe"): 3800,
    ("Middle East", "Latin America"): 12000,
    ("Middle East", "North America"): 10500,
    ("Middle East", "Middle East"): 1200,
}

MODE_LEAD_DAYS = {
    "Air": 18.0,
    "Air Charter": 12.0,
    "Ocean": 48.0,
    "Truck": 22.0,
    "N/A": 30.0,
}

# Destination country → coarse region (for corridor distance).
COUNTRY_REGION = {
    "south africa": "Africa",
    "nigeria": "Africa",
    "côte d'ivoire": "Africa",
    "cote d'ivoire": "Africa",
    "uganda": "Africa",
    "zambia": "Africa",
    "mozambique": "Africa",
    "tanzania": "Africa",
    "kenya": "Africa",
    "rwanda": "Africa",
    "ethiopia": "Africa",
    "ghana": "Africa",
    "cameroon": "Africa",
    "congo": "Africa",
    "democratic republic of the congo": "Africa",
    "zimbabwe": "Africa",
    "malawi": "Africa",
    "botswana": "Africa",
    "namibia": "Africa",
    "lesotho": "Africa",
    "swaziland": "Africa",
    "eswatini": "Africa",
    "burundi": "Africa",
    "benin": "Africa",
    "togo": "Africa",
    "guinea": "Africa",
    "liberia": "Africa",
    "sierra leone": "Africa",
    "senegal": "Africa",
    "angola": "Africa",
    "sudan": "Africa",
    "south sudan": "Africa",
    "vietnam": "Asia Pacific",
    "india": "Asia Pacific",
    "china": "Asia Pacific",
    "myanmar": "Asia Pacific",
    "cambodia": "Asia Pacific",
    "thailand": "Asia Pacific",
    "indonesia": "Asia Pacific",
    "philippines": "Asia Pacific",
    "pakistan": "Asia Pacific",
    "haiti": "Latin America",
    "dominican republic": "Latin America",
    "guyana": "Latin America",
    "guatemala": "Latin America",
    "honduras": "Latin America",
    "el salvador": "Latin America",
    "nicaragua": "Latin America",
    "peru": "Latin America",
    "bolivia": "Latin America",
    "ecuador": "Latin America",
    "colombia": "Latin America",
    "brazil": "Latin America",
    "mexico": "Latin America",
    "united states": "North America",
    "usa": "North America",
    "canada": "North America",
    "ukraine": "Europe",
    "russia": "Europe",
    "kazakhstan": "Asia Pacific",
    "afghanistan": "Middle East",
    "iraq": "Middle East",
    "yemen": "Middle East",
    "jordan": "Middle East",
    "lebanon": "Middle East",
}


def download_bytes(urls: list[str], timeout: int = 60) -> bytes:
    """Try each URL until one succeeds. Raises RuntimeError if all fail."""
    errors: list[str] = []
    for url in urls:
        try:
            req = Request(url, headers={"User-Agent": "SupplyPrescript/1.0 (research)"})
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                errors.append(f"{url}: empty body")
                continue
            return data
        except Exception as exc:  # noqa: BLE001 - collect and keep trying mirrors
            errors.append(f"{url}: {exc}")
    raise RuntimeError("All download URLs failed:\n  - " + "\n  - ".join(errors))


def cache_raw(raw_dir: Path, name: str, urls: list[str], force: bool = False) -> Path:
    """Download once into data/raw/; reuse the cache unless force=True."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / name
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.write_bytes(download_bytes(urls))
    return dest


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_col(str(c)) for c in df.columns]
    # BOM / leading junk on first column (e.g. "id" after utf-8-sig still ok)
    if df.columns[0].startswith("ufeff"):
        df = df.rename(columns={df.columns[0]: df.columns[0].replace("ufeff", "")})
    return df


def _parse_date(value) -> datetime | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {
        "nan",
        "n/a",
        "na",
        "date not captured",
        "date not delivered",
        "pre-pq process",
        "n/a - from rdc",
    }:
        return None
    for fmt in ("%d-%b-%y", "%d-%b-%Y", "%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _to_float(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {
        "nan",
        "n/a",
        "na",
        "see asn",
        "freight included in commodity cost",
        "weight captured separately",
        "invoiced separately",
    }:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _origin_region_from_site(site: str) -> str:
    text = (site or "").lower()
    if any(tok in text for tok in ("india", "hyderabad", "goa", "bangalore", "nashik", "paonta", "cipla", "hetero", "aurobindo", "mylan", "matrix", "strides", "ranbaxy")):
        return "Asia Pacific"
    if any(tok in text for tok in ("japan", "alere", "inverness japan", "korea", "china", "vietnam", "singapore")):
        return "Asia Pacific"
    if any(tok in text for tok in ("germany", "wiesbaden", "ludwigshafen", "uk", "united kingdom", "ireland", "trinity", "france", "italy", "spain", "netherlands", "belgium", "switzerland")):
        return "Europe"
    if any(tok in text for tok in ("usa", "united states", "us ", "canada", "mexico")):
        return "North America"
    if any(tok in text for tok in ("israel", "orgenics", "jordan", "egypt", "uae", "dubai")):
        return "Middle East"
    if "africa" in text or "rdc" in text or "south africa" in text:
        return "Africa"
    return "Asia Pacific"


def _dest_region(country: str) -> str:
    key = (country or "").strip().lower()
    if key in COUNTRY_REGION:
        return COUNTRY_REGION[key]
    # Soft fallbacks
    if any(tok in key for tok in ("congo", "africa")):
        return "Africa"
    return "Africa"


def _sku_from_row(row: pd.Series) -> str:
    product = str(row.get("product_group") or "ITEM").strip().upper() or "ITEM"
    brand = str(row.get("brand") or "GENERIC").strip()
    brand = re.sub(r"[^A-Za-z0-9]+", "-", brand).strip("-").upper() or "GENERIC"
    molecule = str(row.get("molecule_test_type") or row.get("item_description") or "UNK")
    # Keep SKUs short and stable for one-hot encoding.
    token = re.sub(r"[^A-Za-z0-9]+", "-", molecule.split(",")[0])[:28].strip("-").upper()
    return f"{product}-{brand}-{token}"[:64]


def transform_scms(raw_csv: Path | bytes) -> pd.DataFrame:
    """Map USAID SCMS delivery history → shipment training rows."""
    if isinstance(raw_csv, (bytes, bytearray)):
        df = pd.read_csv(io.BytesIO(raw_csv), encoding="utf-8-sig")
    else:
        df = pd.read_csv(raw_csv, encoding="utf-8-sig")
    df = _normalize_columns(df)

    # Column aliases across mirrors / Socrata exports.
    aliases = {
        "unit_price": ["unit_price", "unit_price_usd"],
        "pack_price": ["pack_price", "pack_price_usd"],
        "line_item_quantity": ["line_item_quantity", "line_item_qty"],
        "line_item_value": ["line_item_value"],
        "freight_cost": ["freight_cost", "freight_cost_usd"],
        "weight": ["weight", "weight_kilograms"],
        "manufacturing_site": ["manufacturing_site"],
        "vendor": ["vendor"],
        "country": ["country"],
        "shipment_mode": ["shipment_mode"],
        "product_group": ["product_group"],
        "brand": ["brand"],
        "item_description": ["item_description"],
        "molecule_test_type": ["molecule_test_type"],
        "scheduled_delivery_date": ["scheduled_delivery_date"],
        "delivered_to_client_date": ["delivered_to_client_date"],
        "po_sent_to_vendor_date": ["po_sent_to_vendor_date"],
    }
    resolved: dict[str, str] = {}
    for canon, options in aliases.items():
        for opt in options:
            if opt in df.columns:
                resolved[canon] = opt
                break

    required = ["vendor", "country", "scheduled_delivery_date", "delivered_to_client_date"]
    missing = [c for c in required if c not in resolved]
    if missing:
        raise ValueError(f"SCMS file missing required columns: {missing}; got {list(df.columns)}")

    rows = []
    # First pass: collect lead times where PO→schedule is available (for vendor averages).
    vendor_leads: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        vendor = str(row[resolved["vendor"]]).strip() or "Unknown Vendor"
        po = _parse_date(row[resolved["po_sent_to_vendor_date"]]) if "po_sent_to_vendor_date" in resolved else None
        sched = _parse_date(row[resolved["scheduled_delivery_date"]])
        if po and sched:
            lead = (sched - po).days
            if 1 <= lead <= 400:
                vendor_leads.setdefault(vendor, []).append(float(lead))

    vendor_lead_avg = {
        vendor: float(np.median(vals)) for vendor, vals in vendor_leads.items() if vals
    }

    for _, row in df.iterrows():
        sched = _parse_date(row[resolved["scheduled_delivery_date"]])
        delivered = _parse_date(row[resolved["delivered_to_client_date"]])
        if not sched or not delivered:
            continue

        # Early arrivals count as 0 delay days (model predicts delay risk, not earliness).
        delay_days = max(0.0, float((delivered - sched).days))

        vendor = str(row[resolved["vendor"]]).strip() or "Unknown Vendor"
        country = str(row[resolved["country"]]).strip()
        site = str(row[resolved["manufacturing_site"]]).strip() if "manufacturing_site" in resolved else ""
        origin = _origin_region_from_site(site) if site else "Asia Pacific"
        dest = _dest_region(country)
        distance = float(CORRIDOR_KM.get((origin, dest), 8000))

        mode = str(row[resolved["shipment_mode"]]).strip() if "shipment_mode" in resolved else "N/A"
        if mode in ("", "nan"):
            mode = "N/A"

        po = _parse_date(row[resolved["po_sent_to_vendor_date"]]) if "po_sent_to_vendor_date" in resolved else None
        if po and sched and 1 <= (sched - po).days <= 400:
            lead = float((sched - po).days)
        elif vendor in vendor_lead_avg:
            lead = vendor_lead_avg[vendor]
        else:
            lead = MODE_LEAD_DAYS.get(mode, 30.0)

        qty = _to_float(row[resolved["line_item_quantity"]]) if "line_item_quantity" in resolved else None
        if qty is None or qty <= 0:
            continue
        qty = int(qty)

        unit = _to_float(row[resolved["unit_price"]]) if "unit_price" in resolved else None
        pack = _to_float(row[resolved["pack_price"]]) if "pack_price" in resolved else None
        value = _to_float(row[resolved["line_item_value"]]) if "line_item_value" in resolved else None
        if unit is not None and unit > 0:
            unit_cost = unit
        elif pack is not None and pack > 0:
            unit_cost = pack
        elif value is not None and value > 0:
            unit_cost = value / qty
        else:
            continue

        # Slight distance jitter from shipment weight so long/heavy moves aren't identical.
        weight = _to_float(row[resolved["weight"]]) if "weight" in resolved else None
        if weight and weight > 0:
            distance *= 1.0 + min(weight, 20_000) / 100_000.0

        sku_vals = {
            "product_group": row[resolved["product_group"]] if "product_group" in resolved else row.get("product_group"),
            "brand": row[resolved["brand"]] if "brand" in resolved else row.get("brand"),
            "molecule_test_type": row[resolved["molecule_test_type"]] if "molecule_test_type" in resolved else row.get("molecule_test_type"),
            "item_description": row[resolved["item_description"]] if "item_description" in resolved else row.get("item_description"),
        }

        rows.append(
            {
                "sku": _sku_from_row(pd.Series(sku_vals)),
                "supplier": vendor[:120],
                "origin_region": origin,
                "distance_km": round(distance, 1),
                "historical_avg_lead_time_days": round(max(lead, 3.0), 1),
                "order_quantity": qty,
                "unit_cost_usd": round(float(unit_cost), 4),
                "is_peak_season": bool(sched.month in (11, 12)),
                "actual_delay_days": round(delay_days, 1),
            }
        )

    out = pd.DataFrame(rows, columns=SHIPMENT_COLUMNS)
    if out.empty:
        raise ValueError("SCMS transform produced 0 usable rows")
    return out.reset_index(drop=True)


def transform_c2k(raw_zip_or_csv: Path | bytes) -> pd.DataFrame:
    """Map UCI Cargo 2000 freight events → shipment training rows (~3.9k)."""
    if isinstance(raw_zip_or_csv, (bytes, bytearray)):
        payload = bytes(raw_zip_or_csv)
        if payload[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".csv"))
                df = pd.read_csv(zf.open(name))
        else:
            df = pd.read_csv(io.BytesIO(payload))
    else:
        path = Path(raw_zip_or_csv)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".csv"))
                df = pd.read_csv(zf.open(name))
        else:
            df = pd.read_csv(path)

    df = df.replace("?", np.nan)
    for col in df.columns:
        if col.endswith(("_p", "_e", "_hops")) or col in {"legs", "nr"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Airport/place ids → coarse origin buckets (anonymized in the UCI release).
    place_cols = [c for c in df.columns if c.endswith("_place")]
    place_to_region = {}
    all_places = pd.unique(df[place_cols].astype(str).values.ravel())
    for i, place in enumerate(sorted(p for p in all_places if p not in {"nan", "None"})):
        place_to_region[place] = ["Europe", "Asia Pacific", "North America", "Middle East"][i % 4]

    # Treat frequent first-leg places as "carriers"/suppliers.
    first_place = df.get("o_dep_1_place")
    supplier_map = {}
    if first_place is not None:
        top = first_place.dropna().astype(str).value_counts().head(12).index.tolist()
        for i, place in enumerate(top):
            supplier_map[place] = f"C2K Carrier {place}"

    rows = []
    for _, row in df.iterrows():
        planned = row.get("o_dlv_p")
        actual = row.get("o_dlv_e")
        if pd.isna(planned) or pd.isna(actual):
            continue
        # Durations are minutes in the UCI schema.
        delay_days = max(0.0, (float(actual) - float(planned)) / 1440.0)
        lead_days = max(3.0, float(planned) / 1440.0)

        place = str(row.get("o_dep_1_place")) if not pd.isna(row.get("o_dep_1_place")) else "unknown"
        origin = place_to_region.get(place, "Europe")
        supplier = supplier_map.get(place, f"C2K Carrier {place}" if place != "unknown" else "C2K Carrier UNK")

        hops = row.get("o_hops")
        hops = float(hops) if not pd.isna(hops) else 1.0
        legs = row.get("legs")
        legs = float(legs) if not pd.isna(legs) else 1.0

        # Distance proxy: planned minutes × typical air speed, scaled by hops.
        distance = max(500.0, lead_days * 650.0 * max(hops, 1.0))

        sku = f"C2K-LEG-{int(legs)}H{int(hops)}"
        # Quantity / unit cost are not in C2K — use stable proxies from process shape
        # so the existing feature set still has signal without inventing fake labels.
        order_quantity = int(500 + 400 * legs + 200 * hops)
        unit_cost = round(8.0 + 1.5 * hops + 0.4 * legs, 2)

        # Peak season proxy from process id buckets (no calendar in C2K).
        nr = int(row.get("nr") or 0)
        is_peak = bool((nr % 12) in (10, 11))

        rows.append(
            {
                "sku": sku,
                "supplier": supplier[:120],
                "origin_region": origin,
                "distance_km": round(distance, 1),
                "historical_avg_lead_time_days": round(lead_days, 1),
                "order_quantity": order_quantity,
                "unit_cost_usd": unit_cost,
                "is_peak_season": is_peak,
                "actual_delay_days": round(delay_days, 1),
            }
        )

    out = pd.DataFrame(rows, columns=SHIPMENT_COLUMNS)
    if out.empty:
        raise ValueError("C2K transform produced 0 usable rows")
    return out.reset_index(drop=True)


ADAPTERS = {
    "usaid-scms": {
        "label": "USAID SCMS Delivery History",
        "urls": SCMS_URLS,
        "cache_name": "scms_delivery_history.csv",
        "transform": transform_scms,
        "is_zip": False,
    },
    "uci-c2k": {
        "label": "UCI Cargo 2000 Freight Tracking",
        "urls": C2K_URLS,
        "cache_name": "c2k_freight_tracking.zip",
        "transform": transform_c2k,
        "is_zip": True,
    },
}
