# -*- coding: utf-8 -*-
"""
S3/MinIO object storage service for file upload.
"""
import os
import hashlib
import datetime
import uuid
from io import BytesIO
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.core.config import settings


class S3StorageService:
    """MinIO storage service for uploading files."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        prefix: Optional[str] = None,
        secure: Optional[bool] = None
    ):
        """
        Initialize MinIO storage service.

        Args:
            endpoint: MinIO endpoint URL (without http:// or https://)
            access_key: S3 access key
            secret_key: S3 secret key
            bucket_name: Bucket name
            prefix: Path prefix for uploaded files
            secure: Use HTTPS if True, HTTP if False
        """
        self.endpoint = endpoint or settings.s3_endpoint
        # Remove http:// or https:// from endpoint if present
        if self.endpoint.startswith(("http://", "https://")):
            self.endpoint = self.endpoint.replace("https://", "").replace("http://", "")

        self.access_key = access_key or settings.s3_access_key
        self.secret_key = secret_key or settings.s3_secret_key
        self.bucket_name = bucket_name or settings.s3_bucket_name
        self.prefix = prefix or settings.s3_prefix
        self.secure = secure if secure is not None else settings.s3_secure

        # 初始化MinIO客户端
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )

    def _generate_object_name(
        self,
        original_filename: str,
        subfolder: Optional[str] = None
    ) -> str:
        """
        Generate object name for S3 upload.

        Args:
            original_filename: Original filename
            subfolder: Optional subfolder within prefix

        Returns:
            Object name for S3
        """
        # Get file extension
        _, ext = os.path.splitext(original_filename)
        ext = ext.lstrip('.')

        # Generate unique filename using timestamp and UUID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}.{ext}"

        # Build full object path
        if subfolder:
            return f"{self.prefix}/{subfolder}/{filename}"
        return f"{self.prefix}/{filename}"

    def _generate_object_name_from_text(
        self,
        text: str,
        codec: str,
        subfolder: Optional[str] = None
    ) -> str:
        """
        Generate object name from text content.

        Args:
            text: Text content
            codec: Audio codec (mp3, wav, etc.)
            subfolder: Optional subfolder within prefix

        Returns:
            Object name for S3
        """
        # Generate hash of text for deterministic naming (optional)
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]

        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}_{text_hash}.{codec}"

        # Build full object path
        if subfolder:
            return f"{self.prefix}/{subfolder}/{filename}"
        return f"{self.prefix}/{filename}"

    def upload_bytes(
        self,
        data: bytes,
        object_name: Optional[str] = None,
        content_type: Optional[str] = None,
        subfolder: Optional[str] = None
    ) -> dict:
        """
        Upload bytes data to MinIO.

        Args:
            data: Bytes data to upload
            object_name: Custom object name (auto-generated if None)
            content_type: Content-Type header
            subfolder: Optional subfolder within prefix

        Returns:
            Dict with upload result containing:
                - success: bool
                - url: str (public URL if successful)
                - object_key: str (MinIO object key)
                - error: str (if failed)
        """
        try:
            # Generate object name if not provided
            if object_name is None:
                object_name = self._generate_object_name("upload.bin", subfolder)

            # Upload bytes
            data_stream = BytesIO(data)
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=data_stream,
                length=len(data),
                content_type=content_type
            )

            # 生成访问URL
            protocol = "https" if self.secure else "http"
            public_url = f"{protocol}://{self.endpoint}/{self.bucket_name}/{object_name}"

            return {
                "success": True,
                "url": public_url,
                "object_key": object_name,
                "bucket": self.bucket_name,
                "size": len(data)
            }

        except S3Error as e:
            return {
                "success": False,
                "error": f"MinIO错误: {str(e)}",
                "url": None,
                "object_key": object_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": None,
                "object_key": object_name
            }

    def upload_file(
        self,
        file_path: str,
        object_name: Optional[str] = None,
        content_type: Optional[str] = None,
        subfolder: Optional[str] = None
    ) -> dict:
        """
        Upload local file to MinIO.

        Args:
            file_path: Path to local file
            object_name: Custom object name (auto-generated if None)
            content_type: Content-Type header
            subfolder: Optional subfolder within prefix

        Returns:
            Dict with upload result
        """
        try:
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"文件不存在: {file_path}",
                    "url": None,
                    "object_key": None
                }

            if object_name is None:
                object_name = os.path.basename(file_path)

            # 完整的对象键
            full_object_name = f"{self.prefix}/{object_name}"
            if subfolder:
                full_object_name = f"{self.prefix}/{subfolder}/{object_name}"

            # 上传文件
            self.client.fput_object(
                self.bucket_name,
                full_object_name,
                file_path,
                content_type=content_type
            )

            # 生成访问URL
            protocol = "https" if self.secure else "http"
            public_url = f"{protocol}://{self.endpoint}/{self.bucket_name}/{full_object_name}"

            return {
                "success": True,
                "url": public_url,
                "object_key": full_object_name,
                "bucket": self.bucket_name,
                "file_size": os.path.getsize(file_path)
            }

        except S3Error as e:
            return {
                "success": False,
                "error": f"MinIO错误: {str(e)}",
                "url": None,
                "object_key": full_object_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": None,
                "object_key": full_object_name
            }

    def upload_tts_audio(
        self,
        audio_data: bytes,
        codec: str,
        text: Optional[str] = None,
        subfolder: str = "tts"
    ) -> dict:
        """
        Upload TTS audio data to MinIO with appropriate naming.

        Args:
            audio_data: Audio bytes
            codec: Audio codec (mp3, wav, etc.)
            text: Original text (for naming)
            subfolder: Subfolder for TTS files

        Returns:
            Dict with upload result
        """
        content_type = "audio/mpeg" if codec == "mp3" else "audio/wav"

        if text:
            object_name = self._generate_object_name_from_text(text, codec, subfolder)
        else:
            object_name = self._generate_object_name(f"audio.{codec}", subfolder)

        return self.upload_bytes(audio_data, object_name, content_type)

    def list_buckets(self):
        """列出所有bucket"""
        try:
            buckets = self.client.list_buckets()
            return [bucket.name for bucket in buckets]
        except Exception as e:
            print(f"列出bucket失败: {e}")
            return []

    def list_objects(self, prefix: Optional[str] = None) -> list:
        """
        List objects in bucket.

        Args:
            prefix: Prefix filter (uses service prefix if None)

        Returns:
            List of object info dicts
        """
        try:
            search_prefix = prefix or self.prefix
            objects = self.client.list_objects(
                self.bucket_name,
                prefix=search_prefix,
                recursive=True
            )

            result = []
            protocol = "https" if self.secure else "http"
            for obj in objects:
                result.append({
                    "key": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "url": f"{protocol}://{self.endpoint}/{self.bucket_name}/{obj.object_name}"
                })

            return result

        except S3Error as e:
            print(f"Error listing objects: {e}")
            return []
        except Exception as e:
            print(f"Error listing objects: {e}")
            return []

    def delete_object(self, object_name: str) -> dict:
        """
        Delete an object from MinIO.

        Args:
            object_name: S3 object key

        Returns:
            Dict with delete result
        """
        try:
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            return {"success": True}
        except S3Error as e:
            return {"success": False, "error": f"MinIO错误: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Default service instance
s3_storage = S3StorageService()
