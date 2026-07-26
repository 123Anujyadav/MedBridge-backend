import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class UserBase(BaseModel):
    email: EmailStr
    role: str = Field(default="patient", pattern="^(patient|doctor|admin)$")
    is_active: bool = True
    is_verified: bool = False

class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=100)

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8, max_length=100)
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None

class UserResponse(UserBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
