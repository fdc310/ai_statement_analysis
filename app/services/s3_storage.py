# -*- coding: utf-8 -*-
"""
S3/MinIO object storage service for file upload.
Supports two upload modes: "oss" (MinIO direct) and "api" (POST to upload API).
"""
import os
import hashlib
import datetime
import uuid
import logging
from io import BytesIO
from typing import Optional

import httpx
from minio import Minio
from minio.error import S3Error

from app.core.config import settings

logger = logging.getLogger(__name__)


class S3StorageService:
    """Storage service supporting MinIO direct upload and POST API upload."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        prefix: Optional[str] = None,
        secure: Optional[bool] = None,
        upload_mode: Optional[str] = None,
        upload_api_url: Optional[str] = None,
        public_url: Optional[str] = None
    ):
        self.endpoint = endpoint or settings.s3_endpoint
        if self.endpoint.startswith(("http://", "https://")):
            self.endpoint = self.endpoint.replace("https://", "").replace("http://", "")

        self.access_key = access_key or settings.s3_access_key
        self.secret_key = secret_key or settings.s3_secret_key
        self.bucket_name = bucket_name or settings.s3_bucket_name
        self.prefix = prefix or settings.s3_prefix
        self.secure = secure if secure is not None else settings.s3_secure

        # 上传模式: "oss" = MinIO直传, "api" = POST接口上传
        self.upload_mode = upload_mode or settings.upload_mode
        self.upload_api_url = upload_api_url or settings.upload_api_url

        # 访问域名（阿里云OSS操作域名和访问域名不同）
        # 例如: https://liaoyu-public.oss-cn-beijing.aliyuncs.com
        self.public_url = public_url or settings.s3_public_url

        # 初始化MinIO客户端（oss模式或需要list/delete等操作时使用）
        # 阿里云OSS需要指定region，避免MinIO自动调用GetBucketLocation被403
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
            region="oss-cn-beijing"
        )

    # ==================== 内部工具方法 ====================

    def _generate_object_name(
        self,
        original_filename: str,
        subfolder: Optional[str] = None
    ) -> str:
        _, ext = os.path.splitext(original_filename)
        ext = ext.lstrip('.')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}.{ext}"
        if subfolder:
            return f"{self.prefix}/{subfolder}/{filename}"
        return f"{self.prefix}/{filename}"

    def _generate_object_name_from_text(
        self,
        text: str,
        codec: str,
        subfolder: Optional[str] = None
    ) -> str:
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}_{text_hash}.{codec}"
        if subfolder:
            return f"{self.prefix}/{subfolder}/{filename}"
        return f"{self.prefix}/{filename}"

    def _generate_oss_id(self) -> str:
        """生成19位纯数字ID"""
        return str(uuid.uuid4().int)[:19]

    def _build_public_url(self, object_name: str) -> str:
        """生成访问URL，优先使用配置的公共访问域名"""
        if self.public_url:
            return f"{self.public_url.rstrip('/')}/{object_name}"
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.bucket_name}.{self.endpoint}/{object_name}"

    def _format_result(self, success: bool, url: str = None, file_name: str = None,
                        oss_id: str = None, error: str = None) -> dict:
        """统一返回格式"""
        if success:
            return {
                "success": True,
                "url": url,
                "fileName": file_name,
                "ossId": oss_id
            }
        return {
            "success": False,
            "error": error,
            "url": None,
            "fileName": None,
            "ossId": None
        }

    # ==================== POST API 上传 ====================

    def _post_upload_bytes(
        self,
        data: bytes,
        file_name: str = "upload.bin",
        content_type: Optional[str] = None
    ) -> dict:
        """通过POST接口上传bytes数据。"""
        try:
            files = {"file": (file_name, BytesIO(data), content_type or "application/octet-stream")}
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.upload_api_url, files=files)
                response.raise_for_status()

            result = response.json()
            if result.get("code") == 200:
                data_body = result.get("data", {})
                return self._format_result(
                    True,
                    url=data_body.get("url"),
                    file_name=data_body.get("fileName"),
                    oss_id=data_body.get("ossId")
                )
            else:
                return self._format_result(False, error=f"上传接口返回错误: {result.get('msg')}")

        except Exception as e:
            logger.exception("POST上传失败")
            return self._format_result(False, error=str(e))

    def _post_upload_file(self, file_path: str) -> dict:
        """通过POST接口上传本地文件。"""
        try:
            if not os.path.exists(file_path):
                return self._format_result(False, error=f"文件不存在: {file_path}")

            file_name = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                files = {"file": (file_name, f)}
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(self.upload_api_url, files=files)
                    response.raise_for_status()

            result = response.json()
            if result.get("code") == 200:
                data_body = result.get("data", {})
                return self._format_result(
                    True,
                    url=data_body.get("url"),
                    file_name=data_body.get("fileName"),
                    oss_id=data_body.get("ossId")
                )
            else:
                return self._format_result(False, error=f"上传接口返回错误: {result.get('msg')}")

        except Exception as e:
            logger.exception("POST上传文件失败")
            return self._format_result(False, error=str(e))

    # ==================== MinIO OSS 直传 ====================

    def _oss_upload_bytes(
        self,
        data: bytes,
        object_name: Optional[str] = None,
        content_type: Optional[str] = None,
        subfolder: Optional[str] = None
    ) -> dict:
        """通过MinIO SDK直传bytes到OSS。"""
        try:
            if object_name is None:
                object_name = self._generate_object_name("upload.bin", subfolder)

            data_stream = BytesIO(data)
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=data_stream,
                length=len(data),
                content_type=content_type
            )

            public_url = self._build_public_url(object_name)
            file_name = os.path.basename(object_name)

            return self._format_result(True, url=public_url, file_name=file_name, oss_id=self._generate_oss_id())

        except S3Error as e:
            return self._format_result(False, error=f"MinIO错误: {str(e)}")
        except Exception as e:
            return self._format_result(False, error=str(e))

    def _oss_upload_file(
        self,
        file_path: str,
        object_name: Optional[str] = None,
        content_type: Optional[str] = None,
        subfolder: Optional[str] = None
    ) -> dict:
        """通过MinIO SDK直传本地文件到OSS。"""
        try:
            if not os.path.exists(file_path):
                return self._format_result(False, error=f"文件不存在: {file_path}")

            if object_name is None:
                object_name = os.path.basename(file_path)

            full_object_name = f"{self.prefix}/{object_name}"
            if subfolder:
                full_object_name = f"{self.prefix}/{subfolder}/{object_name}"

            self.client.fput_object(
                self.bucket_name,
                full_object_name,
                file_path,
                content_type=content_type
            )

            public_url = self._build_public_url(full_object_name)
            file_name = os.path.basename(file_path)

            return self._format_result(True, url=public_url, file_name=file_name, oss_id=self._generate_oss_id())

        except S3Error as e:
            return self._format_result(False, error=f"MinIO错误: {str(e)}")
        except Exception as e:
            return self._format_result(False, error=str(e))

    # ==================== 统一对外接口（根据upload_mode自动选择） ====================

    def upload_bytes(
        self,
        data: bytes,
        object_name: Optional[str] = None,
        content_type: Optional[str] = None,
        subfolder: Optional[str] = None
    ) -> dict:
        """
        上传bytes数据，根据upload_mode自动选择上传方式。

        Args:
            data: 文件字节数据
            object_name: 自定义对象名（oss模式使用）
            content_type: Content-Type
            subfolder: 子目录

        Returns:
            Dict with url, fileName, ossId
        """
        if self.upload_mode == "api":
            file_name = os.path.basename(object_name) if object_name else "upload.bin"
            return self._post_upload_bytes(data, file_name, content_type)
        else:
            return self._oss_upload_bytes(data, object_name, content_type, subfolder)

    def upload_file(
        self,
        file_path: str,
        object_name: Optional[str] = None,
        content_type: Optional[str] = None,
        subfolder: Optional[str] = None
    ) -> dict:
        """
        上传本地文件，根据upload_mode自动选择上传方式。

        Args:
            file_path: 本地文件路径
            object_name: 自定义对象名（oss模式使用）
            content_type: Content-Type
            subfolder: 子目录

        Returns:
            Dict with url, fileName, ossId
        """
        if self.upload_mode == "api":
            return self._post_upload_file(file_path)
        else:
            return self._oss_upload_file(file_path, object_name, content_type, subfolder)

    def upload_tts_audio(
        self,
        audio_data: bytes,
        codec: str,
        text: Optional[str] = None,
        subfolder: str = "tts"
    ) -> dict:
        """
        上传TTS音频数据，根据upload_mode自动选择上传方式。

        Args:
            audio_data: 音频字节数据
            codec: 音频编码 (mp3, wav, etc.)
            text: 原始文本（用于命名）
            subfolder: 子目录

        Returns:
            Dict with url, fileName, ossId
        """
        content_type = "audio/mpeg" if codec == "mp3" else "audio/wav"

        if text:
            object_name = self._generate_object_name_from_text(text, codec, subfolder)
        else:
            object_name = self._generate_object_name(f"audio.{codec}", subfolder)

        if self.upload_mode == "api":
            file_name = os.path.basename(object_name)
            return self._post_upload_bytes(audio_data, file_name, content_type)
        else:
            return self._oss_upload_bytes(audio_data, object_name, content_type)

    # ==================== OSS管理操作（始终走MinIO） ====================

    def list_buckets(self):
        """列出所有bucket"""
        try:
            buckets = self.client.list_buckets()
            return [bucket.name for bucket in buckets]
        except Exception as e:
            print(f"列出bucket失败: {e}")
            return []

    def list_objects(self, prefix: Optional[str] = None) -> list:
        """List objects in bucket."""
        try:
            search_prefix = prefix or self.prefix
            objects = self.client.list_objects(
                self.bucket_name,
                prefix=search_prefix,
                recursive=True
            )

            result = []
            for obj in objects:
                result.append({
                    "url": self._build_public_url(obj.object_name),
                    "fileName": os.path.basename(obj.object_name),
                    "ossId": self._generate_oss_id()
                })

            return result

        except S3Error as e:
            print(f"Error listing objects: {e}")
            return []
        except Exception as e:
            print(f"Error listing objects: {e}")
            return []

    def delete_object(self, object_name: str) -> dict:
        """Delete an object from MinIO."""
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
