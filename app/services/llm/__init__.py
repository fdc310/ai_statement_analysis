from app.services.llm.factory import LLMFactory
from app.core.config import settings

def get_default_llm():
    """获取系统默认配置的大模型实例（动态读取后台配置）"""
    # 动态导入避免循环依赖
    from app.api.v1.endpoints.admin import MOCK_DB
    
    config = MOCK_DB["llm_config"]
    provider = config.get("provider", "tencent")
    provider_settings = config.get("providers", {}).get(provider, {})
    
    if provider == "tencent":
        # 如果后台没填，回退到环境变量
        secret_id = provider_settings.get("secret_id") or settings.tencent_secret_id
        secret_key = provider_settings.get("secret_key") or settings.tencent_secret_key
        model = provider_settings.get("model") or settings.hunyuan_model
        
        return LLMFactory.get_service(
            "tencent", 
            secret_id=secret_id,
            secret_key=secret_key,
            model=model
        )
        
    elif provider in ["openai", "deepseek"]:
        api_key = provider_settings.get("api_key")
        base_url = provider_settings.get("base_url")
        model = provider_settings.get("model")
        
        # 兼容性处理，如果未填用占位符防止崩溃
        if not api_key:
            import logging
            logging.warning(f"API key for {provider} is empty, this call will fail.")
            
        return LLMFactory.get_service(
            provider, 
            api_key=api_key or "sk-dummy", 
            base_url=base_url, 
            model=model
        )
        
    # 默认回退
    return LLMFactory.get_service("tencent", model=settings.hunyuan_model)
