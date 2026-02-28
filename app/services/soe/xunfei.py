import json
import asyncio
import websockets
import base64
from app.services.soe.base import BaseSOEService
from app.services.xunfei_util import create_xunfei_url

class XunfeiSOEService(BaseSOEService):
    """
    科大讯飞语音评测 (SOE) 服务 - WebSocket 接口 (ISE)
    """
    def __init__(self, app_id: str, api_key: str, api_secret: str):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_url = "wss://ise-api.xfyun.cn/v2/open-ise"
        
    async def evaluate(self, audio_path: str, text: str, **kwargs) -> dict:
        def run_ws():
            ws_url = create_xunfei_url(self.api_url, self.api_key, self.api_secret)
            
            # 读取音频数据
            with open(audio_path, 'rb') as f:
                audio_data = f.read()

            try:
                ws = websockets.sync.client.connect(ws_url)
                
                # SOE 是完整的 XML 结果
                result_xml = ""
                error_msg = None
                
                # 第一帧：发送业务参数
                frame_size = 8000
                status = 0 # 0: 第一帧
                
                for i in range(0, len(audio_data), frame_size):
                    chunk = audio_data[i:i+frame_size]
                    if i + frame_size >= len(audio_data):
                        status = 2 # 最后一帧
                    elif status == 0:
                        status = 0
                        
                    data_dict = {
                        "common": {
                            "app_id": self.app_id
                        },
                        "business": {
                            "category": "read_sentence", # 评测题型：句篇(默认)
                            "rstcd": "utf8",
                            "group": "xhyy",
                            "sub_asn": "ise",
                            "ent": "cn_vip", # cn_vip表示中文
                            "tte": "utf-8",
                            "text": '\uFEFF' + text # 文本要求添加BOM头
                        },
                        "data": {
                            "status": status,
                            "data": base64.b64encode(chunk).decode('utf-8')
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
                    if data:
                        base64_data = data.get("data", "")
                        if base64_data:
                            result_xml += base64.b64decode(base64_data).decode('utf-8')
                            
                    if data.get("status") == 2:
                        break
                        
                ws.close()
                return result_xml, error_msg
            except Exception as e:
                return "", str(e)

        loop = asyncio.get_event_loop()
        xml_res, err = await loop.run_in_executor(None, run_ws)
        
        if err:
            return {"success": False, "error": f"Xunfei SOE Error: {err}"}
            
        # 简单将 XML 解析或提取得分 (实际项目中需要解析讯飞的 XML 取出 pron_accuracy)
        # 讯飞返回的 XML 里包含 <total_score> 等字段。为了跟腾讯的数据结构保持一致，这里用正则做一个兼容转换。
        import re
        total_score_match = re.search(r'total_score="([^"]+)"', xml_res)
        score = float(total_score_match.group(1)) * 20 if total_score_match else 0.0 # 讯飞往往是5分制，转为100分
        
        return {
            "success": True,
            "scores": {
                "pron_accuracy": score,
                "pron_fluency": score, # 粗略兼容
                "pron_completion": score
            },
            "statistics": {},
            "low_score_words": [],
            "raw_xml": xml_res
        }
