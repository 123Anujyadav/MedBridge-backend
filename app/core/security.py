import bcrypt
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Union
import jwt
from app.core.config import settings


def _unique_claims() -> dict:
    """
    Per-issuance identity claims.

    `exp` has one-second resolution, so two tokens minted for the same subject
    within the same second used to encode to byte-identical strings. Because the
    refresh flow blacklists the presented token by its own string, an identical
    replacement came back already revoked and the session died on the next
    refresh. A random `jti` guarantees every issued token is distinct.
    """
    return {
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_urlsafe(16),
    }

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Checks if a plain text password matches a stored hashed password using bcrypt.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    """
    Generates a secure bcrypt hash from a plain text password.
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def create_access_token(
    subject: Union[str, Any], expires_delta: Union[timedelta, None] = None
) -> str:
    """
    Generates a short-lived access JWT token.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "access",
                 **_unique_claims()}
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt

def create_refresh_token(
    subject: Union[str, Any], expires_delta: Union[timedelta, None] = None
) -> str:
    """
    Generates a long-lived refresh JWT token.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh",
                 **_unique_claims()}
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt

def decode_token(token: str) -> dict:
    """
    Decodes and validates a JWT token. Raises PyJWT exceptions on invalid state.
    """
    return jwt.decode(
        token,
        settings.JWT_SECRET,
        algorithms=[settings.JWT_ALGORITHM]
    )
