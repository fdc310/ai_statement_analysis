import json
import asyncio
import websockets
import _thread as thread
from app.services.asr.base import BaseASRService
from app.services.xunfei_util import create_xunfei_url

class XunfeiASRService(BaseASRService):
    """
    科大讯飞语音听写 (ASR) 服务 - WebSocket 接口
    """
    def __init__(self, app_id: str, api_key: str, api_secret: str):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        # 语音听写流式接口
        self.api_url = "wss://iat-api.xfyun.cn/v2/iat"
        
    async def recognize(self, audio_path: str, **kwargs) -> dict:
        """
        通过 WebSocket 发送音频并获取 ASR 结果
        """
        # 使用 asyncio.to_thread 防止阻塞主循环，因为 websockets(同步库) 需要独立线程/进程
        # 这里为了快速集成和稳定，我们使用一个简单的闭包包装同步 websocket 逻辑
        def run_ws():
            ws_url = create_xunfei_url(self.api_url, self.api_key, self.api_secret)
            
            result_text = []
            error_msg = None
            
            # 读取音频数据
            with open(audio_path, 'rb') as f:
                audio_data = f.read()

            try:
                # 为了简化代码，我们在短连接中一次性发送 (使用帧大小切片也是可以的)
                ws = websockets.sync.client.connect(ws_url)
                
                # 第一帧：发送业务参数
                frame_size = 8000
                status = 0 # 0: 第一帧
                
                # 开始切片发送
                for i in range(0, len(audio_data), frame_size):
                    chunk = audio_data[i:i+frame_size]
                    if i + frame_size >= len(audio_data):
                        status = 2 # 最后一帧
                    elif status == 0:
                        status = 0 # 第一帧之后变成1
                        
                    data_dict = {
                        "common": {
                            "app_id": self.app_id
                        },
                        "business": {
                            "domain": "iat",
                            "language": kwargs.get("language", "zh_cn"),
                            "accent": "mandarin",
                            "vinfo": 1,
                            "vad_eos": 10000
                        },
                        "data": {
                            "status": status,
                            "format": "audio/L16;rate=16000",
                            "encoding": "raw",
                            "audio": __import__('base64').b64encode(chunk).decode('utf-8')
                        }
                    }
                    if status != 0:
                        data_dict.pop("common")
                        data_dict.pop("business")
                        
                    ws.send(json.dumps(data_dict))
                    
                    if status == 0:
                        status = 1
                
                # 接收结果
                while True:
                    res = ws.recv()
                    res_dict = json.loads(res)
                    if res_dict.get("code") != 0:
                        error_msg = res_dict.get("message", "Unknown error")
                        break
                        
                    data = res_dict.get("data", {})
                    result_info = data.get("result", {})
                    ws_list = result_info.get("ws", [])
                    for w in ws_list:
                        for cw in w.get("cw", []):
                            result_text.append(cw.get("w", ""))
                            
                    if data.get("status") == 2:
                        break
                        
                ws.close()
                return "".join(result_text), error_msg
            except Exception as e:
                return "", str(e)

        loop = asyncio.get_event_loop()
        text, err = await loop.run_in_executor(None, run_ws)
        
        if err:
            return {"success": False, "error": f"Xunfei ASR Error: {err}"}
            
        return {
            "success": True,
            "text": text,
            # 讯飞的动态时间戳较复杂，需要提取，这里暂时留空或按需解析
            "word_info_list": [] 
        }
