import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.base import BaseRepository
from app.models.user import User

@pytest.mark.asyncio
async def test_base_repository_crud_and_soft_delete(db: AsyncSession):
    """
    Tests BaseRepository CRUD operations, soft delete, and soft remove filtering.
    """
    repo = BaseRepository(User)

    # 1. Create User
    user = User(
        email="repo_test@aronofy.com",
        hashed_password="hashed_secret_pass",
        role="patient"
    )
    db.add(user)
    await db.flush()
    user_id = user.id

    # 2. Get user
    retrieved = await repo.get(db, user_id)
    assert retrieved is not None
    assert retrieved.email == "repo_test@aronofy.com"

    # 3. Get multi
    users_list = await repo.get_multi(db, skip=0, limit=10)
    assert len(users_list) >= 1

    # 4. Soft remove user
    soft_deleted = await repo.soft_remove(db, id=user_id)
    assert soft_deleted is not None
    assert soft_deleted.deleted_at is not None
    assert soft_deleted.is_deleted is True

    # 5. Verify get excludes soft deleted by default
    hidden = await repo.get(db, user_id)
    assert hidden is None

    # 6. Verify get with include_deleted=True returns record
    with_deleted = await repo.get(db, user_id, include_deleted=True)
    assert with_deleted is not None
    assert with_deleted.id == user_id

    # 7. Permanent remove
    await repo.remove(db, id=user_id)
    permanently_deleted = await repo.get(db, user_id, include_deleted=True)
    assert permanently_deleted is None
