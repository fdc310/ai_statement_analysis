from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

router = APIRouter()

# 临时使用内存模拟数据库存储，未来接入 SQLite/MySQL/Redis
MOCK_DB = {
    "llm_config": {
        "provider": "tencent",  # "tencent" | "openai" | "deepseek"
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
            }
        }
    }
}

class LLMProviderConfig(BaseModel):
    provider: str
    providers: Dict[str, dict]

@router.get("/config/llm", response_model=LLMProviderConfig)
async def get_llm_config():
    """获取当前 LLM 提供商配置"""
    return MOCK_DB["llm_config"]

@router.post("/config/llm")
async def update_llm_config(config: LLMProviderConfig):
    """更新 LLM 提供商配置"""
    MOCK_DB["llm_config"] = config.dict()
    # TODO: 这里未来可以通过 Event Bus 通知其他 worker 热重载配置
    return {"success": True, "message": "Configuration updated successfully"}
