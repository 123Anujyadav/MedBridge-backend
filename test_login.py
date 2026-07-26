import asyncio
import logging
from app.core.database import AsyncSessionLocal
from app.schemas.auth import LoginRequest
from app.services.auth import auth_service
from app.core.redis import get_redis

async def main():
    async with AsyncSessionLocal() as db:
        redis = await anext(get_redis())
        for role, email, password in [
            ("Patient", "patient@example.com", "patient123"),
            ("Doctor", "doctor@example.com", "doctor123"),
            ("Admin", "admin@example.com", "admin123"),
        ]:
            try:
                tokens = await auth_service.login(
                    db,
                    LoginRequest(email=email, password=password),
                    redis
                )
                print(f"[SUCCESS] {role} Login verified: {email}")
            except Exception as e:
                print(f"[ERROR] {role} Login failed ({email}): {e}")

if __name__ == "__main__":
    asyncio.run(main())
