from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseSOEService(ABC):
    """
    语音评测(SOE)服务的抽象基类
    """
    
    @abstractmethod
    async def evaluate(self, audio_path: str, text: str, **kwargs) -> Dict[str, Any]:
        """
        评测音频文件发音
        
        Args:
            audio_path: 音频文件路径
            text: 参考文本 (如果是自由说则可能不需要)
            **kwargs: 额外参数 (如 score_coeff, language等)
            
        Returns:
            Dict 包含:
                - success: bool
                - scores: dict (包含 pron_accuracy, pron_fluency 等)
                - statistics: dict (可选统计信息)
                - low_score_words: list (可选低分词列表)
                - speech_rate: float (可选语速)
                - error: str (失败时的错误信息)
        """
        pass
