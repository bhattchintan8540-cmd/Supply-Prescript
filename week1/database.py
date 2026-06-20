from __future__ import annotations

from sqlalchemy import create_engine
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


def init_db() -> None:
    """Create tables if they don't exist yet. Called from main.py on startup
    and from the pytest fixtures so tests don't need a migration tool."""
    from . import models  # noqa: F401  (import registers the tables on Base)

    Base.metadata.create_all(bind=engine)
