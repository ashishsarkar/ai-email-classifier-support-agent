"""
Configuration management using Pydantic Settings.

This module loads configuration from environment variables and .env file.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    ENV: str = "development"

    # LLM Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    VLLM_BASE_URL: Optional[str] = None  # e.g. "http://litellm:4000/v1"

    # Fallback LLM Configuration (Claude via LiteLLM)
    ANTHROPIC_API_KEY: Optional[str] = None
    FALLBACK_MODEL: str = "claude-sonnet-4-6-20250514"
    FALLBACK_BASE_URL: Optional[str] = None  # e.g. "http://litellm:4000/v1" for Claude via LiteLLM

    # Circuit Breaker Configuration
    CB_FAILURE_THRESHOLD: int = 3          # Consecutive failures before opening circuit
    CB_COOLDOWN_SECONDS: int = 60          # Seconds before trying primary again (half-open)
    CB_HALF_OPEN_MAX_CALLS: int = 1        # Max test calls in half-open state

    # LangSmith Tracing Configuration
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "customer-support-email-agent"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_ENABLED: bool = False

    # Email Configuration
    IMAP_SERVER: str = "imap.gmail.com"
    IMAP_PORT: int = 993
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    EMAIL_ADDRESS: str
    EMAIL_PASSWORD: str
    EMAIL_FROM_NAME: str = "Customer Support"

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./customer_support.db"

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # Email Processing Configuration
    EMAIL_CHECK_INTERVAL: int = 300  # seconds
    MAX_EMAILS_PER_BATCH: int = 10
    EMAIL_RESPONSE_TIMEOUT: int = 30  # seconds

    # Vector Store Configuration
    VECTOR_STORE_PATH: str = "./data/vectors"

    class Config:
        """Pydantic config."""

        env_file = ".env"
        case_sensitive = True


# Create a global settings instance
settings = Settings()
