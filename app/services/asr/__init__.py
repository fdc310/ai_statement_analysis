from app.services.asr.factory import ASRFactory
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.config import SystemConfig

def get_default_asr():
    """获取系统默认配置的 ASR 实例（从 SQLite 数据库实时读取）"""
    db = SessionLocal()
    try:
        config_record = db.query(SystemConfig).filter(SystemConfig.key == "asr_config").first()
        if config_record and config_record.value:
            config = config_record.value
        else:
            config = {"provider": "tencent", "providers": {}}
    except Exception as e:
        import logging
        logging.error(f"Failed to read ASR config from DB: {e}")
        config = {"provider": "tencent", "providers": {}}
    finally:
        db.close()
        
    provider = config.get("provider", "tencent")
    provider_settings = config.get("providers", {}).get(provider, {})
    
    if provider == "tencent":
        secret_id = provider_settings.get("secret_id") or settings.tencent_secret_id
        secret_key = provider_settings.get("secret_key") or settings.tencent_secret_key
        appid = provider_settings.get("appid")
        return ASRFactory.get_service("tencent", secret_id=secret_id, secret_key=secret_key, appid=appid)
        
    elif provider == "xunfei":
        app_id = provider_settings.get("app_id")
        api_key = provider_settings.get("api_key")
        api_secret = provider_settings.get("api_secret")
        return ASRFactory.get_service("xunfei", app_id=app_id, api_key=api_key, api_secret=api_secret)
        
    return ASRFactory.get_service("tencent", secret_id=settings.tencent_secret_id, secret_key=settings.tencent_secret_key)
