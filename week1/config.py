"""
Settings for SupplyPrescript.

Nothing fancy here - just pulling values from the environment so the
same code works against a local sqlite file (fast to iterate on) and
a real Postgres instance (docker-compose, staging, whatever) without
touching a single line of application code.
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Default to a local sqlite file so `pytest` and quick manual runs work
# out of the box with zero setup. Point DATABASE_URL at Postgres for
# anything beyond a laptop demo, e.g.
#   postgresql+psycopg2://sp_user:sp_pass@localhost:5432/supplyprescript
DATABASE_URL = os.getenv(
    "DATABASE_URL", f"sqlite:///{ROOT_DIR / 'data' / 'supplyprescript.db'}"
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "data" / "delay_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", ROOT_DIR / "data" / "metrics.json"))

# Business constraints the solver enforces by default. A real deployment
# would probably pull these from a settings table so ops can tune them
# without a redeploy, but env vars are good enough for the portfolio build.
DEFAULT_BUDGET_USD = float(os.getenv("SP_DEFAULT_BUDGET", 100_000))
DEFAULT_MAX_DELAY_DAYS = int(os.getenv("SP_DEFAULT_MAX_DELAY_DAYS", 14))

# Operational delay semantics for the optimizer.
# When partial fulfillment is NOT useful (default), production cannot start
# until the last unit arrives — so the constraint is on makespan
# (max delay across used channels), not quantity-weighted average delay.
# Set SP_PARTIAL_FULFILLMENT_USEFUL=1 only when arriving units create
# usable business value before the full order is complete.
PARTIAL_FULFILLMENT_USEFUL = os.getenv("SP_PARTIAL_FULFILLMENT_USEFUL", "0") in {
    "1",
    "true",
    "True",
    "yes",
    "YES",
}

# Minimum fraction of the order that must travel on channels whose
# resulting delay is within the SLA (max_acceptable_delay_days). Useful
# even when partial fulfillment is allowed — e.g. "at least 70% on time."
DEFAULT_MIN_ON_TIME_FRACTION = float(os.getenv("SP_MIN_ON_TIME_FRACTION", 0.0))

# How far predicted cost can drift from actual cost before we bother
# retraining (see week4/retrain.py). 15% felt like a reasonable "don't
# retrain over noise" threshold - tune as more decisions accumulate.
RETRAIN_DRIFT_THRESHOLD = float(os.getenv("SP_RETRAIN_DRIFT", 0.15))
