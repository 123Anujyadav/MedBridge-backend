import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_db():
    try:
        conn = await asyncpg.connect("postgresql://postgres:postgrespassword@localhost:5432/postgres")
        db_exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = 'aronofy_db'")
        if not db_exists:
            await conn.execute("CREATE DATABASE aronofy_db")
            logger.info("Created PostgreSQL database: aronofy_db")
        else:
            logger.info("PostgreSQL database aronofy_db already exists")
        await conn.close()
    except Exception as e:
        logger.error(f"Error creating database: {e}")

if __name__ == "__main__":
    asyncio.run(create_db())
