import logging
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TrialSight Intelligence"
    app_env: str = "development"
    jwt_secret: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    database_url: str = "sqlite:///./data/app.db"
    upload_dir: str = "./data/uploads"
    redis_url: str | None = None

    llm_provider: str = "groq"
    groq_api_key: str | None = None
    # Groq retired the Llama 3.x models on 2026-08-16; gpt-oss-120b is its recommended replacement.
    groq_model: str = "openai/gpt-oss-120b"
    # Extra completion tokens reserved for a reasoning model's hidden thinking.
    groq_reasoning_token_headroom: int = 1024

    dense_vector_size: int = 768

    # Amazon Bedrock (used when llm_provider / embedding_provider == "bedrock").
    # Credentials come from the standard AWS chain (EC2 instance role in production).
    aws_region: str | None = None
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    embedding_provider: str = "hash"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_embedding_dim: int = 512

    # Serve the static frontend from the API process (single-origin deploy, no CORS needed).
    serve_frontend: bool = False
    frontend_dir: str = "./frontend"

    max_file_size_bytes: int = 10 * 1024 * 1024
    max_uploads_per_day: int = 5
    max_files_per_upload: int = 1
    max_queries_per_minute: int = 4
    max_queries_per_day: int = 25
    demo_global_queries_per_minute: int = 8
    demo_global_queries_per_day: int = 250
    max_input_tokens: int = 1200
    max_generation_tokens: int = 500

    groq_requests_per_minute: int = 20
    groq_requests_per_day: int = 500
    groq_tokens_per_minute: int = 4000
    groq_tokens_per_day: int = 100000
    groq_max_concurrent_requests: int = 2
    groq_answer_scoring_enabled: bool = False

    top_k_dense: int = 8
    top_k_keyword: int = 8
    top_k_final: int = 6
    query_cache_ttl_seconds: int = 1800

    frontend_origins: str = Field(default="http://localhost:3000,http://localhost:5173")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.frontend_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.app_env != "development" and s.jwt_secret == "change-this-in-production":
        logging.getLogger(__name__).warning(
            "JWT_SECRET is set to the default value. Set a strong, unique secret for production."
        )
    return s
