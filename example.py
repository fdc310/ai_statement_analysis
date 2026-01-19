"""
Example script demonstrating API usage with AES encrypted requests.
"""
import requests
import json
import time
import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

BASE_URL = "http://localhost:8000/api/v1"

# These should match your .env configuration
AES_KEY = "your_aes_key_32bytes_here_!!!!"
AES_IV = "your_aes_iv_16bytes"


class AESEncryptor:
    """AES encryption helper for API requests."""

    def __init__(self, key: str, iv: str):
        self.key = self._normalize_key(key)
        self.iv = self._normalize_iv(iv)

    def _normalize_key(self, key: str) -> bytes:
        key_bytes = key.encode("utf-8")
        if len(key_bytes) < 32:
            key_bytes = hashlib.sha256(key_bytes).digest()
        return key_bytes[:32]

    def _normalize_iv(self, iv: str) -> bytes:
        iv_bytes = iv.encode("utf-8")
        if len(iv_bytes) < 16:
            iv_bytes = hashlib.md5(iv_bytes).digest()
        return iv_bytes[:16]

    def encrypt(self, data: dict) -> str:
        """Encrypt request data with timestamp."""
        payload = {
            "data": data,
            "timestamp": int(time.time())
        }
        plaintext = json.dumps(payload, ensure_ascii=False)

        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        padded_data = pad(plaintext.encode("utf-8"), AES.block_size)
        encrypted = cipher.encrypt(padded_data)
        return base64.b64encode(encrypted).decode("utf-8")


# Initialize encryptor
encryptor = AESEncryptor(AES_KEY, AES_IV)


def evaluate_from_url(audio_url: str, custom_prompt: str = None):
    """Evaluate speech from audio URL."""
    # Build request data
    data = {
        "audio_url": audio_url,
        "language": "zh",
        "voice_format": "mp3"
    }
    if custom_prompt:
        data["custom_prompt"] = custom_prompt

    # Encrypt the request data
    encrypted_data = encryptor.encrypt(data)

    response = requests.post(
        f"{BASE_URL}/evaluation/analyze",
        json={"encrypted_data": encrypted_data}
    )
    return response.json()


def evaluate_from_file(file_path: str, custom_prompt: str = None):
    """Evaluate speech from uploaded file."""
    # Build options data
    data = {
        "language": "zh"
    }
    if custom_prompt:
        data["custom_prompt"] = custom_prompt

    # Encrypt the options
    encrypted_data = encryptor.encrypt(data)

    with open(file_path, "rb") as f:
        files = {"file": f}
        form_data = {"encrypted_data": encrypted_data}

        response = requests.post(
            f"{BASE_URL}/evaluation/analyze/upload",
            files=files,
            data=form_data
        )
    return response.json()


def main():
    print("=== AI Statement Analysis API Example ===\n")

    # Check health
    print("1. Checking API health...")
    try:
        response = requests.get(f"{BASE_URL}/health/health")
        print(f"   Status: {response.json()}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
        return

    # Evaluate speech from URL
    print("2. Evaluating speech from URL...")
    try:
        audio_url = "https://example.com/speech.mp3"
        result = evaluate_from_url(
            audio_url,
            custom_prompt="请特别关注演讲者的论证逻辑和语言表达能力"
        )
        print(f"   Success: {result.get('success')}")
        if result.get('success'):
            print(f"   Speech Text: {result.get('speech_text', '')[:100]}...")
            print(f"   Scores: {json.dumps(result.get('speech_scores'), indent=2, ensure_ascii=False)}")
            print(f"\n   Evaluation Report:\n{result.get('evaluation_report', '')}")
        else:
            print(f"   Error: {result.get('error')}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
