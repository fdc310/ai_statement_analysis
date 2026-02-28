import json
import httpx
from typing import Optional, AsyncGenerator

from app.services.llm.base import BaseLLMService


class OpenAICompatibleService(BaseLLMService):
    """
    OpenAI 兼容接口的大模型服务 (DeepSeek, ChatGPT, Qwen, Moonshot 等通用)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo"
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> dict:
        """调用兼容 OpenAI 格式的 API 接口"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                return self._parse_chat_result(data)
            except Exception as e:
                import logging
                logging.error(f"OpenAICompatibleService chat error: {str(e)}")
                raise

    def _parse_chat_result(self, data: dict) -> dict:
        """解析 OpenAI 格式的响应"""
        choices = data.get("choices", [])
        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            
        usage = data.get("usage", {})
        
        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            "raw_response": data
        }
