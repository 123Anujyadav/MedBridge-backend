import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_tables():
    conn = await asyncpg.connect("postgresql://postgres:postgrespassword@localhost:5432/aronofy_db")
    tables = await conn.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
    )
    table_names = [t["table_name"] for t in tables]
    logger.info(f"PostgreSQL Tables ({len(table_names)} Total):")
    for name in table_names:
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {name}")
        logger.info(f"  - {name}: {count} rows")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(verify_tables())
