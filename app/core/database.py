import logging
import sys
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.core.config import settings

logger = logging.getLogger(__name__)

# Create Async Engine for PostgreSQL using asyncpg driver
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
    pool_size=20,
    max_overflow=10
)

# Create Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an async database session
    and rolls back changes in case of an unhandled exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Transaction error, rolling back: {str(e)}")
            await session.rollback()
            raise e
        finally:
            await session.close()
