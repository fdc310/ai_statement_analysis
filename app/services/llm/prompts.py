"""
Prompt management module.
This will eventually be replaced by database queries for a management backend.
"""

# We migrate the prompt building functions from hunyuan.py here as pure text generation functions.
# To keep this step focused, we will first create a function that takes a prompt and calls the factory LLM.

import json
from app.services.llm import get_default_llm

async def generate_json_with_llm(prompt: str) -> dict:
    """
    通用的大模型调用接口，专门用于请求 JSON 格式数据。
    它会通过 Factory 获取当前配置的模型并进行调用。
    """
    llm = get_default_llm()
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 强制模型输出 JSON 的额外系统提示可以视需求添加
    # messages.insert(0, {"role": "system", "content": "你是一个有用的助手，请只输出纯 JSON 数据，不要包含 markdown 标记。"})
    
    response = await llm.chat(messages, temperature=0.7)
    content = response.get("content", "").strip()
    
    # 清理可能存在的 markdown 标记
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
        
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        import logging
        logging.error(f"Failed to parse LLM JSON response: {content}")
        # 如果解析失败，返回一个带错误的字典
        return {"error": "Invalid JSON from LLM", "raw_content": content}
