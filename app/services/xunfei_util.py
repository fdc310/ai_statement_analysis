import hmac
import hashlib
import base64
from datetime import datetime
from time import mktime
from wsgiref.handlers import format_date_time
from urllib.parse import urlencode, urlparse

def create_xunfei_url(api_url: str, api_key: str, api_secret: str) -> str:
    """生成科大讯飞 WebSocket 请求的鉴权 URL"""
    url_parts = urlparse(api_url)
    host = url_parts.netloc
    path = url_parts.path

    # 生成RFC1123格式的时间戳
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))

    # 拼接签名字符串
    signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"

    # 进行hmac-sha256加密
    signature_sha = hmac.new(
        api_secret.encode('utf-8'),
        signature_origin.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()

    signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

    # 拼接authorization
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

    # 将鉴权参数拼接到URL中
    v = {
        "authorization": authorization,
        "date": date,
        "host": host
    }
    return api_url + '?' + urlencode(v)
