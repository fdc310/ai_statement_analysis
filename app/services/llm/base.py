from abc import ABC, abstractmethod

class BaseLLMService(ABC):
    """
    大模型服务的抽象基类 (Abstract Base Class for LLM services)
    任何新的模型提供商（混元、DeepSeek、OpenAI）都需要实现这个接口。
    """
    
    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> dict:
        """
        统一聊天接口。
        
        Args:
            messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
            temperature: 随机性
            top_p: 多样性
            stream: 是否流式返回
            
        Returns:
            dict: {
                "content": "模型生成的文本",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "raw_response": <原始响应对象>
            }
        """
        pass
