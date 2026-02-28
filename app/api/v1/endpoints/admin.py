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