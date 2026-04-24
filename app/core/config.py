"""
Application configuration management.
"""
import os
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "AI Statement Analysis API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Tencent Cloud
    tencent_secret_id: str = os.getenv("TENCENT_SECRET_ID", "")
    tencent_secret_key: str = os.getenv("TENCENT_SECRET_KEY", "")
    tencent_appid: str = os.getenv("TENCENT_APPID", "")

    # AES Encryption
    aes_key: str = os.getenv("AES_KEY", "default_key_16b!")
    aes_iv: str = os.getenv("AES_IV", "default_iv_16b!!")

    # Request expiration (seconds) - encrypted requests older than this are rejected
    request_expire_seconds: int = int(os.getenv("REQUEST_EXPIRE_SECONDS", "300"))

    # API Server Config
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # LLM Provider Config (openai / tencent / anthropic)
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "120"))  # Timeout in seconds for LLM requests

    # Tencent Hunyuan Model Config
    tencent_model: str = os.getenv("TENCENT_MODEL", "hunyuan-turbo")
    tencent_multimodal_model: str = os.getenv("TENCENT_MULTIMODAL_MODEL", "hunyuan-multimodal")

    # OpenAI Model Config
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_multimodal_model: str = os.getenv("OPENAI_MULTIMODAL_MODEL", "gpt-4o-audio-preview")

    # Anthropic Model Config
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    anthropic_multimodal_model: str = os.getenv("ANTHROPIC_MULTIMODAL_MODEL", "claude-sonnet-4-20250514")

    # S3/MinIO Object Storage Config
    s3_endpoint: str = os.getenv("S3_ENDPOINT", "")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "")
    s3_bucket_name: str = os.getenv("S3_BUCKET_NAME", "")
    s3_prefix: str = os.getenv("S3_PREFIX", "")
    s3_secure: bool = os.getenv("S3_SECURE", "false").lower() == "true"
    s3_public_url: str = os.getenv("S3_PUBLIC_URL", "")

    # Upload Config
    # upload_mode: "oss" = MinIO直传OSS, "api" = POST接口上传
    upload_mode: str = os.getenv("UPLOAD_MODE", "api")
    upload_api_url: str = os.getenv("UPLOAD_API_URL", "")

    # Thread Pool
    sdk_thread_pool_size: int = int(os.getenv("SDK_THREAD_POOL_SIZE", "20"))

    # Task Queue
    task_cleanup_interval: int = int(os.getenv("TASK_CLEANUP_INTERVAL", "3600"))
    task_max_age: int = int(os.getenv("TASK_MAX_AGE", "86400"))

    # Monitoring
    monitoring_enabled: bool = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
    monitoring_retention_days: int = int(os.getenv("MONITORING_RETENTION_DAYS", "30"))

    # Streaming
    stream_max_session_duration: int = int(os.getenv("STREAM_MAX_SESSION_DURATION", "300"))
    stream_audio_buffer_size: int = int(os.getenv("STREAM_AUDIO_BUFFER_SIZE", "1048576"))

    # Chat Session
    chat_session_ttl: int = int(os.getenv("CHAT_SESSION_TTL", "3600"))  # 1 hour

    # LLM Pricing (per 1M tokens, USD)
    tencent_input_price: float = float(os.getenv("TENCENT_INPUT_PRICE", "0.5"))
    tencent_output_price: float = float(os.getenv("TENCENT_OUTPUT_PRICE", "1.0"))
    openai_input_price: float = float(os.getenv("OPENAI_INPUT_PRICE", "2.5"))
    openai_output_price: float = float(os.getenv("OPENAI_OUTPUT_PRICE", "10.0"))
    anthropic_input_price: float = float(os.getenv("ANTHROPIC_INPUT_PRICE", "3.0"))
    anthropic_output_price: float = float(os.getenv("ANTHROPIC_OUTPUT_PRICE", "15.0"))

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
