from app.services.asr.base import BaseASRService

class ASRFactory:
    """ASR 服务工厂"""
    
    @staticmethod
    def get_service(provider: str, **kwargs) -> BaseASRService:
        provider = provider.lower()
        
        if provider == "tencent":
            # 兼容现有腾讯的调用，我们需要给腾讯ASR包一层适配器
            from app.services.tencent.asr import ASRService
            
            class TencentASRAdapter(BaseASRService):
                def __init__(self, **kw):
                    self.client = ASRService(
                        secret_id=kw.get("secret_id"),
                        secret_key=kw.get("secret_key"),
                        appid=kw.get("appid")
                    )
                
                async def recognize(self, audio_path: str, **kw) -> dict:
                    return await self.client.recognize(audio_path)
                    
            return TencentASRAdapter(**kwargs)
            
        elif provider == "xunfei":
            # 讯飞的实现在这里
            from app.services.asr.xunfei import XunfeiASRService
            return XunfeiASRService(
                app_id=kwargs.get("app_id"),
                api_key=kwargs.get("api_key"),
                api_secret=kwargs.get("api_secret")
            )
            
        else:
            raise ValueError(f"Unsupported ASR provider: {provider}")
