"""
Application configuration management.
"""
import os
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

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
