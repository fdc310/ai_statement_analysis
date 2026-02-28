from app.services.soe.base import BaseSOEService

class SOEFactory:
    """SOE 服务工厂"""
    
    @staticmethod
    def get_service(provider: str, **kwargs) -> BaseSOEService:
        provider = provider.lower()
        
        if provider == "tencent":
            # 兼容现有腾讯的调用，给腾讯SOE包一层适配器
            from app.services.tencent.soe import SOEService
            
            class TencentSOEAdapter(BaseSOEService):
                def __init__(self, **kw):
                    self.client = SOEService(
                        secret_id=kw.get("secret_id"),
                        secret_key=kw.get("secret_key"),
                        appid=kw.get("appid")
                    )
                
                async def evaluate(self, audio_path: str, text: str, **kw) -> dict:
                    return await self.client.evaluate(
                        audio_path=audio_path,
                        text=text,
                        score_coeff=kw.get("score_coeff", 1.0),
                        language=kw.get("language", "zh")
                    )
                    
            return TencentSOEAdapter(**kwargs)
            
        elif provider == "xunfei":
            # 讯飞的实现在这里
            from app.services.soe.xunfei import XunfeiSOEService
            return XunfeiSOEService(
                app_id=kwargs.get("app_id"),
                api_key=kwargs.get("api_key"),
                api_secret=kwargs.get("api_secret")
            )
            
        else:
            raise ValueError(f"Unsupported SOE provider: {provider}")
