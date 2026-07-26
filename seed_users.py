import asyncio
import logging
from sqlalchemy import select
from app.core.database import AsyncSessionLocal, engine
from app.db.base import Base
from app.core.security import get_password_hash
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def seed_users():
    # Ensure tables exist
    async with engine.begin() as conn:
        # NOTE: schema is owned by Alembic (`alembic upgrade head`), not by this script.
        # `create_all` here is a convenience for throwaway local/demo databases only.
        # It does NOT write an alembic_version row, so a database built this way will
        # fail a later `alembic upgrade head` with "relation already exists". If you use
        # it, follow up with `alembic stamp head`.
        await conn.run_sync(Base.metadata.create_all)
        
    async with AsyncSessionLocal() as db:
        # 1. Patient User
        patient_email = "patient@example.com"
        result = await db.execute(select(User).filter(User.email == patient_email))
        patient_user = result.scalars().first()

        if not patient_user:
            patient_user = User(
                email=patient_email,
                hashed_password=get_password_hash("patient123"),
                role="patient",
                is_active=True,
                is_verified=True,
            )
            db.add(patient_user)
            await db.flush()

            patient_profile = Patient(
                id=patient_user.id,
                first_name="Jane",
                last_name="Doe",
                phone="+1 (555) 123-4567",
                date_of_birth="1992-05-15",
                gender="female",
                blood_type="O+",
                weight=64.5,
                height=168.0,
                allergies=["Penicillin"],
                chronic_conditions=["Asthma"],
                emergency_contact={"name": "John Doe", "relationship": "Spouse", "phone": "+1 (555) 987-6543"},
                address="123 Health Ave",
                city="Chicago",
                state="IL",
                health_score=88,
            )
            db.add(patient_profile)
            logger.info("Created Patient user: patient@example.com / patient123")
        else:
            patient_user.hashed_password = get_password_hash("patient123")
            patient_user.is_active = True
            patient_user.is_verified = True
            logger.info("Updated Patient user: patient@example.com / patient123")

        # 2. Doctor User
        doctor_email = "doctor@example.com"
        result = await db.execute(select(User).filter(User.email == doctor_email))
        doctor_user = result.scalars().first()

        if not doctor_user:
            doctor_user = User(
                email=doctor_email,
                hashed_password=get_password_hash("doctor123"),
                role="doctor",
                is_active=True,
                is_verified=True,
            )
            db.add(doctor_user)
            await db.flush()

            doctor_profile = Doctor(
                id=doctor_user.id,
                first_name="Rachel",
                last_name="Goldberg",
                phone="+1 (555) 321-7654",
                specialty="Cardiology",
                license_number="MD-987654-IL",
                hospital_name="St. Jude General Hospital",
                years_of_experience=12,
                consultation_fee=150.0,
                bio="Senior Cardiologist specializing in preventive heart health.",
                rating=4.9,
                total_patients=124,
                total_cases=310,
                availability="available",
                verification_status="verified",
            )
            db.add(doctor_profile)
            logger.info("Created Doctor user: doctor@example.com / doctor123")
        else:
            doctor_user.hashed_password = get_password_hash("doctor123")
            doctor_user.is_active = True
            doctor_user.is_verified = True
            logger.info("Updated Doctor user: doctor@example.com / doctor123")

        # 3. Admin User
        admin_email = "admin@example.com"
        result = await db.execute(select(User).filter(User.email == admin_email))
        admin_user = result.scalars().first()

        if not admin_user:
            admin_user = User(
                email=admin_email,
                hashed_password=get_password_hash("admin123"),
                role="admin",
                is_active=True,
                is_verified=True,
            )
            db.add(admin_user)
            logger.info("Created Admin user: admin@example.com / admin123")
        else:
            admin_user.hashed_password = get_password_hash("admin123")
            admin_user.is_active = True
            admin_user.is_verified = True
            logger.info("Updated Admin user: admin@example.com / admin123")

        await db.commit()
        logger.info("Database seeding completed successfully!")

if __name__ == "__main__":
    asyncio.run(seed_users())
