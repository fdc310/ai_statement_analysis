from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.security import verify_admin_credentials
from app.core.database import get_db
from app.models.config import SystemConfig
from pydantic import BaseModel
from typing import Optional, Dict

router = APIRouter()

class LLMProviderConfig(BaseModel):
    provider: str
    providers: Dict[str, dict]

# 默认配置，在数据库为空时返回
DEFAULT_LLM_CONFIG = {
    "provider": "tencent",
    "providers": {
        "tencent": {
            "secret_id": "",
            "secret_key": "",
            "model": "hunyuan-turbo"
        },
        "deepseek": {
            "api_key": "",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat"
        },
        "openai": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o"
        }
    }
}

@router.get("/config/llm", response_model=LLMProviderConfig)
async def get_llm_config(
    admin: str = Depends(verify_admin_credentials),
    db: Session = Depends(get_db)
):
    """获取当前 LLM 提供商配置"""
    config_record = db.query(SystemConfig).filter(SystemConfig.key == "llm_config").first()
    if config_record and config_record.value:
        return config_record.value
    return DEFAULT_LLM_CONFIG

@router.post("/config/llm")
async def update_llm_config(
    config: LLMProviderConfig, 
    admin: str = Depends(verify_admin_credentials),
    db: Session = Depends(get_db)
):
    """更新 LLM 提供商配置"""
    config_record = db.query(SystemConfig).filter(SystemConfig.key == "llm_config").first()
    
    if not config_record:
        config_record = SystemConfig(key="llm_config", value=config.dict(), description="LLM Provider Routing Configuration")
        db.add(config_record)
    else:
        # SQLAlchemy needs to know the JSON field has been modified
        config_record.value = config.dict()
        
    db.commit()
    return {"success": True, "message": "Configuration saved to database"}
class ASRProviderConfig(BaseModel):
    provider: str
    providers: Dict[str, dict]

class SOEProviderConfig(BaseModel):
    provider: str
    providers: Dict[str, dict]

DEFAULT_ASR_CONFIG = {
    "provider": "tencent",
    "providers": {
        "tencent": {"secret_id": "", "secret_key": "", "appid": ""},
        "xunfei": {"app_id": "", "api_key": "", "api_secret": ""}
    }
}

DEFAULT_SOE_CONFIG = {
    "provider": "tencent",
    "providers": {
        "tencent": {"secret_id": "", "secret_key": "", "appid": ""},
        "xunfei": {"app_id": "", "api_key": "", "api_secret": ""}
    }
}

@router.get("/config/asr", response_model=ASRProviderConfig)
async def get_asr_config(admin: str = Depends(verify_admin_credentials), db: Session = Depends(get_db)):
    config_record = db.query(SystemConfig).filter(SystemConfig.key == "asr_config").first()
    return config_record.value if config_record and config_record.value else DEFAULT_ASR_CONFIG

@router.post("/config/asr")
async def update_asr_config(config: ASRProviderConfig, admin: str = Depends(verify_admin_credentials), db: Session = Depends(get_db)):
    config_record = db.query(SystemConfig).filter(SystemConfig.key == "asr_config").first()
    if not config_record:
        config_record = SystemConfig(key="asr_config", value=config.dict(), description="ASR Provider Configuration")
        db.add(config_record)
    else:
        config_record.value = config.dict()
    db.commit()
    return {"success": True}

@router.get("/config/soe", response_model=SOEProviderConfig)
async def get_soe_config(admin: str = Depends(verify_admin_credentials), db: Session = Depends(get_db)):
    config_record = db.query(SystemConfig).filter(SystemConfig.key == "soe_config").first()
    return config_record.value if config_record and config_record.value else DEFAULT_SOE_CONFIG

@router.post("/config/soe")
async def update_soe_config(config: SOEProviderConfig, admin: str = Depends(verify_admin_credentials), db: Session = Depends(get_db)):
    config_record = db.query(SystemConfig).filter(SystemConfig.key == "soe_config").first()
    if not config_record:
        config_record = SystemConfig(key="soe_config", value=config.dict(), description="SOE Provider Configuration")
        db.add(config_record)
    else:
        config_record.value = config.dict()
    db.commit()
    return {"success": True}
