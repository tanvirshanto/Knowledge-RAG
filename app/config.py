from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Local embeddings (BGE-M3)
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 32

    # Qdrant Cloud
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "medical_textbooks"

    # Google Gemini API
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.1

    # Context-injected chunking
    chunk_size: int = 1000
    chunk_overlap: int = 300

    # Retrieval
    retrieval_top_k: int = 20

    # Ingestion
    temp_upload_dir: str = "tmp_uploads"

    # Supabase
    supabase_url: str
    supabase_service_key: str

    # JWT Authentication
    jwt_secret: str
    access_token_expire_minutes: int = 30

    # Admin seed
    seed_admin_username: str = "admin"
    seed_admin_password: str = "Admin@1234"
    seed_admin_role: str = "maintainer"

    # Worker
    worker_poll_interval_seconds: int = 60

    # Upload limits
    max_upload_size_mb: int = 50

    # Retry
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    retry_max_delay_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
