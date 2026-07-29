import sys
from pathlib import Path

# this file now lives at the project root itself (not inside tests/),
# since it needs to apply to week1/tests, week2/tests, week3/tests, and
# week4/tests alike - pytest walks up from each test file looking for
# conftest.py, so one copy here covers every week folder.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import os
import pytest

# point the app at a throwaway sqlite file before anything imports
# week1.config, so tests never touch data/supplyprescript.db
TEST_DB_PATH = ROOT / "data" / "test_supplyprescript.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

from week1.database import init_db, engine, Base  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _fresh_test_db():
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
