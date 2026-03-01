from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseASRService(ABC):
    """
    语音识别(ASR)服务的抽象基类
    """
    
    @abstractmethod
    async def recognize(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        识别音频文件内容
        
        Args:
            audio_path: 音频文件路径
            **kwargs: 额外参数
            
        Returns:
            Dict 包含:
                - success: bool
                - text: str (识别文本)
                - word_info_list: list (词级时间戳，可选)
                - error: str (失败时的错误信息)
        """
        pass
