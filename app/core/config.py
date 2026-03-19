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

    # Hunyuan Model Config (hunyuan-turbo = Tencent HY 2.0 Instruct)
    hunyuan_model: str = os.getenv("HUNYUAN_MODEL", "hunyuan-turbo")
    hunyuan_timeout: int = int(os.getenv("HUNYUAN_TIMEOUT", "120"))  # Timeout in seconds for LLM requests

    # OpenAI Model Config
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # LLM Provider Config (hunyuan or openai)
    llm_provider: str = os.getenv("LLM_PROVIDER", "hunyuan")

    # S3/MinIO Object Storage Config
    s3_endpoint: str = os.getenv("S3_ENDPOINT", "")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "")
    s3_bucket_name: str = os.getenv("S3_BUCKET_NAME", "")
    s3_prefix: str = os.getenv("S3_PREFIX", "")
    s3_secure: bool = os.getenv("S3_SECURE", "false").lower() == "true"

    # Upload Config
    # upload_mode: "oss" = MinIO直传OSS, "api" = POST接口上传
    upload_mode: str = os.getenv("UPLOAD_MODE", "api")
    upload_api_url: str = os.getenv("UPLOAD_API_URL", "")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
