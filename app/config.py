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

    # Command Code Provider API (OpenAI-compatible)
    command_code_api_key: str
    command_code_base_url: str = "https://api.commandcode.ai/provider/v1"
    command_code_model: str = "deepseek/deepseek-v4-flash"
    llm_temperature: float = 0.1

    # Context-injected chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    retrieval_top_k: int = 6

    # Ingestion
    temp_upload_dir: str = "tmp_uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
