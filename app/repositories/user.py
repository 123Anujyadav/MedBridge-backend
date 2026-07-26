from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.repositories.base import BaseRepository
from app.schemas.user import UserCreate, UserUpdate
from app.schemas.patient import PatientCreate
from app.schemas.doctor import DoctorCreate

class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        """
        Retrieves a user by their email address.
        """
        result = await db.execute(select(User).where(User.email == email))
        return result.scalars().first()

    async def get_by_supabase_id(
        self, db: AsyncSession, supabase_user_id: str
    ) -> Optional[User]:
        """Retrieves the account linked to a Supabase identity."""
        result = await db.execute(
            select(User).where(User.supabase_user_id == supabase_user_id)
        )
        return result.scalars().first()

    async def create_patient_user(
        self, db: AsyncSession, email: str, hashed_password: str,
        profile_in: PatientCreate, supabase_user_id: Optional[str] = None
    ) -> User:
        """
        Registers a User account and creates a Patient profile atomically.

        `supabase_user_id` links the account to its identity when Supabase is
        the identity provider; it stays null under the built-in provider.
        """
        # Create User credential record
        user = User(email=email, hashed_password=hashed_password, role="patient",
                    supabase_user_id=supabase_user_id)
        db.add(user)
        await db.flush()  # Populate user.id

        # Create Patient clinical profile linked to User.id
        profile_data = profile_in.model_dump()
        patient = Patient(id=user.id, **profile_data)
        db.add(patient)
        await db.flush()
        
        return user

    async def create_doctor_user(
        self, db: AsyncSession, email: str, hashed_password: str,
        profile_in: DoctorCreate, supabase_user_id: Optional[str] = None
    ) -> User:
        """
        Registers a User account and creates a Doctor profile atomically.

        The doctor profile is created with the default `pending` verification
        status, so a new clinician cannot practise until an administrator
        approves them.
        """
        # Create User credential record
        user = User(email=email, hashed_password=hashed_password, role="doctor",
                    supabase_user_id=supabase_user_id)
        db.add(user)
        await db.flush()  # Populate user.id

        # Create Doctor clinical profile linked to User.id
        profile_data = profile_in.model_dump()
        doctor = Doctor(id=user.id, **profile_data)
        db.add(doctor)
        await db.flush()
        
        return user

user_repository = UserRepository(User)
