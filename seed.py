import logging

from app.config import get_settings
from auth.security import hash_password
from repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)


async def seed_admin_user() -> None:
    settings = get_settings()
    user_repo = UserRepository()

    if not user_repo.table_exists():
        logger.error(
            "Users table does not exist. Run the SQL in this file's docstring "
            "in the Supabase SQL Editor before starting the application."
        )
        return

    if user_repo.exists_by_username(settings.seed_admin_username):
        logger.info("Admin user '%s' already exists, skipping seed.", settings.seed_admin_username)
        return

    pw_hash = hash_password(settings.seed_admin_password)
    user_repo.create(
        username=settings.seed_admin_username,
        password_hash=pw_hash,
        role=settings.seed_admin_role,
    )
    logger.info(
        "Seeded admin user: %s (role: %s)",
        settings.seed_admin_username,
        settings.seed_admin_role,
    )
