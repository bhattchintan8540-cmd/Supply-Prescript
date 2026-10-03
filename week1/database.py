from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from .config import DATABASE_URL

# check_same_thread only matters for sqlite - Postgres just ignores the
# connect_args entirely if we're not sqlite, so no need to branch on it.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_session():
    """FastAPI dependency - yields a session, always closes it after."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# Columns added after the first local sqlite file was created.
# create_all does not alter an existing table.
_DECISION_COLUMNS = {
    "shipment_features_json": "TEXT",
    "no_action_cost_usd": "FLOAT",
    "resolved_at": "DATETIME",
}


def ensure_missing_columns(bind=engine) -> None:
    """Add nullable decision columns when an older sqlite file is missing them."""
    url = str(bind.url)
    if not url.startswith("sqlite"):
        return
    insp = inspect(bind)
    if "decisions" not in insp.get_table_names():
        return
    existing = {col["name"] for col in insp.get_columns("decisions")}
    with bind.begin() as conn:
        for name, ddl in _DECISION_COLUMNS.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE decisions ADD COLUMN {name} {ddl}"))


def init_db() -> None:
    """Create tables if they don't exist yet. Called from main.py on startup
    and from the pytest fixtures so tests don't need a migration tool."""
    from . import models  # noqa: F401  (import registers the tables on Base)

    Base.metadata.create_all(bind=engine)
    ensure_missing_columns(engine)
