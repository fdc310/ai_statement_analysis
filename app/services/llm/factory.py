from app.services.llm.base import BaseLLMService
from app.services.llm.openai_compatible import OpenAICompatibleService
# Note: For Hunyuan, you would import the Hunyuan class here.

class LLMFactory:
    """
    大模型服务工厂。
    用于根据配置（环境变量或数据库）动态实例化具体的大模型客户端。
    """
    
    @staticmethod
    def get_service(provider: str, **kwargs) -> BaseLLMService:
        """
        获取对应供应商的大模型服务。
        
        Args:
            provider: "tencent", "openai", "deepseek", "qwen" 等
            kwargs: 传递给具体客户端的配置参数 (api_key, base_url, model 等)
        """
        provider = provider.lower()
        
        # 为了兼容你现在的代码架构，我们暂且把腾讯混元作为一个特殊的分支
        if provider == "tencent":
            # 引入现有的 hunyuan service (暂时保留它在原目录，或者你可以把它移到 llm 目录下)
            from app.services.tencent.hunyuan import HunyuanService
            return HunyuanService(
                secret_id=kwargs.get("secret_id"),
                secret_key=kwargs.get("secret_key"),
                model=kwargs.get("model")
            )
            
        elif provider in ["openai", "deepseek", "qwen", "moonshot"]:
            # 对于兼容 OpenAI 格式的模型（绝大多数现代大模型）
            api_key = kwargs.get("api_key")
            base_url = kwargs.get("base_url")
            model = kwargs.get("model")
            
            if not api_key:
                raise ValueError(f"Provider {provider} requires api_key")
                
            return OpenAICompatibleService(
                api_key=api_key,
                base_url=base_url or "https://api.openai.com/v1",
                model=model
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
