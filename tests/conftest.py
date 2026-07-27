import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# The identity provider is pinned before any application module is imported.
#
# Otherwise the suite would follow whatever `.env` happens to say: switching a
# developer's machine to Supabase would send the entire test run at a live
# project, and hundreds of tests asserting built-in token behaviour would fail
# for reasons that have nothing to do with the code under test. The Supabase
# integration is covered by `test_supabase_auth.py`, which opts in explicitly
# and stubs the provider.
os.environ["AUTH_PROVIDER"] = "local"

import asyncio
from typing import AsyncGenerator, Generator
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.core.config import settings
settings.ENVIRONMENT = "testing"


from app.core.database import get_db
from app.core.redis import get_redis
from app.db.base import Base
from app.main import app

# Use file-based SQLite for sharing across testing contexts
TEST_DATABASE_URL = "sqlite+aiosqlite:///test.db"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False
)

TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """
    Creates an instance of the event loop for the duration of the test session.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session", autouse=True)
async def setup_test_db():
    """
    Initializes clean database schema structure on test postgres DB at session start
    and tears down tables on completion.
    """
    async with test_engine.begin() as conn:
        # In a real environment, you might run alembic migrations instead,
        # but metadata.create_all is robust for isolated testing.
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()

_active_session: AsyncSession | None = None
"""
The session the test in flight is using — the same one the app is handed.

Read by `login_payload`, which has to see rows a setup fixture has only
flushed. A second session would open its own transaction and find nothing.
"""


@pytest.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields an isolated database session per test case, wrapping in transaction rollback.
    """
    global _active_session
    async with TestSessionLocal() as session:
        _active_session = session
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            _active_session = None
            await session.close()

@pytest.fixture
async def client(db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Provides an async HTTP client for executing queries against test endpoint routes.
    Overrides session injection targets to map tests database session.
    """
class MockRedis:
    """
    In-memory Mock Redis database for test state tracking.
    """
    def __init__(self):
        self.store = {}

    async def get(self, name: str):
        return self.store.get(name)

    async def set(self, name: str, value: str, ex: int = None):
        self.store[name] = str(value)
        return True

    async def delete(self, name: str):
        self.store.pop(name, None)
        return True

    async def ping(self):
        return True

    async def close(self):
        return True

@pytest.fixture
def mock_redis() -> MockRedis:
    """
    Yields the persistent MockRedis store for test verification.
    """
    return MockRedis()

@pytest.fixture
async def client(db: AsyncSession, mock_redis: MockRedis) -> AsyncGenerator[AsyncClient, None]:
    """
    Provides an async HTTP client for executing queries against test endpoint routes.
    Overrides session and Redis dependencies.
    """
    async def override_get_db():
        yield db

    async def override_get_redis():
        yield mock_redis

    # Apply Dependency Injection overrides
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis] = override_get_redis

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as ac:
        yield ac

    # Tear down overrides to restore standard dependencies
    app.dependency_overrides.clear()



async def login_payload(email: str, password: str = "password123") -> dict:
    """
    Build a sign-in body, adding the Doctor ID when the account is a clinician.

    A doctor signs in with three factors, so a two-factor body is refused — which
    is the behaviour under test elsewhere, not something to work around here.
    Tests create clinicians directly through the model, where the Doctor ID is
    generated by the column default, so the value has to be read back rather
    than assumed.

    Read through the running test's own session, because setup fixtures
    routinely flush without committing and a second session would not see the
    clinician at all.

    Non-doctor accounts get exactly the body they always sent.
    """
    from sqlalchemy import select

    from app.models.doctor import Doctor
    from app.models.user import User

    payload = {"email": email, "password": password}
    lookup = (
        select(Doctor.doctor_code)
        .join(User, User.id == Doctor.id)
        .where(User.email == email)
    )

    if _active_session is not None:
        doctor_code = await _active_session.scalar(lookup)
    else:
        async with TestSessionLocal() as session:
            doctor_code = await session.scalar(lookup)

    if doctor_code:
        payload["doctor_id"] = doctor_code
    return payload
