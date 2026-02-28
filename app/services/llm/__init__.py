from app.services.llm.factory import LLMFactory

# Provide a default singleton instance or a way to get the default configured LLM
# In the future, this config can be fetched from a database per-request
from app.core.config import settings

def get_default_llm():
    """获取系统默认配置的大模型实例"""
    # 这里可以根据未来新增的环境变量来决定是腾讯还是DeepSeek
    # 例如: llm_provider = os.getenv("LLM_PROVIDER", "tencent")
    llm_provider = getattr(settings, "llm_provider", "tencent")
    
    if llm_provider == "tencent":
        return LLMFactory.get_service("tencent", model=settings.hunyuan_model)
    elif llm_provider == "deepseek":
        # 假设 settings 里有 deepseek_api_key 等配置
        return LLMFactory.get_service(
            "deepseek", 
            api_key=getattr(settings, "deepseek_api_key", ""), 
            base_url="https://api.deepseek.com", 
            model="deepseek-chat"
        )
    return LLMFactory.get_service("tencent", model=settings.hunyuan_model)
