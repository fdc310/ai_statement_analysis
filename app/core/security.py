"""
Security utilities including AES encryption/decryption.
"""
import base64
import hashlib
import time
import json
from typing import Optional, Tuple
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

from app.core.config import settings


class AESService:
    """AES-256-CBC encryption service for API authentication."""

    def __init__(self, key: Optional[str] = None, iv: Optional[str] = None):
        self.key = self._normalize_key(key or settings.aes_key)
        self.iv = self._normalize_iv(iv or settings.aes_iv)

    def _normalize_key(self, key: str) -> bytes:
        """Normalize key to 32 bytes (AES-256)."""
        key_bytes = key.encode("utf-8")
        if len(key_bytes) < 32:
            key_bytes = hashlib.sha256(key_bytes).digest()
        return key_bytes[:32]

    def _normalize_iv(self, iv: str) -> bytes:
        """Normalize IV to 16 bytes."""
        iv_bytes = iv.encode("utf-8")
        if len(iv_bytes) < 16:
            iv_bytes = hashlib.md5(iv_bytes).digest()
        return iv_bytes[:16]

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext using AES-256-CBC."""
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        padded_data = pad(plaintext.encode("utf-8"), AES.block_size)
        encrypted = cipher.encrypt(padded_data)
        return base64.b64encode(encrypted).decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext using AES-256-CBC."""
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        encrypted_data = base64.b64decode(ciphertext)
        decrypted = unpad(cipher.decrypt(encrypted_data), AES.block_size)
        return decrypted.decode("utf-8")

    def generate_signature(self, timestamp: Optional[int] = None) -> Tuple[str, int]:
        """
        Generate AES encrypted signature for API authentication.

        Args:
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            Tuple of (signature, timestamp)
        """
        if timestamp is None:
            timestamp = int(time.time())
        payload = {"timestamp": timestamp}
        signature = self.encrypt(json.dumps(payload))
        return signature, timestamp

    def verify_signature(
        self,
        signature: str,
        max_age_seconds: int = 300
    ) -> Tuple[bool, str]:
        """
        Verify AES encrypted signature.

        Args:
            signature: Base64 encoded AES encrypted signature
            max_age_seconds: Maximum allowed age of signature in seconds

        Returns:
            Tuple of (success, message)
        """
        try:
            decrypted = self.decrypt(signature)
            payload = json.loads(decrypted)

            timestamp = payload.get("timestamp")
            if timestamp is None:
                return False, "Invalid signature format"

            current_time = int(time.time())
            if abs(current_time - timestamp) > max_age_seconds:
                return False, "Signature expired"

            return True, "Valid"

        except Exception as e:
            return False, f"Signature verification failed: {str(e)}"


# Singleton instance
aes_service = AESService()
