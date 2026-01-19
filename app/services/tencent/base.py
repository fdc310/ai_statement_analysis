"""
Tencent Cloud base client configuration with async support.
"""
import logging
import sys
from typing import Optional

from tencentcloud.common import credential, retry
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

from app.core.config import settings


# Configure retry logger
retry_logger = logging.getLogger("tencentcloud.retry")
retry_logger.setLevel(logging.WARNING)
retry_logger.addHandler(logging.StreamHandler(sys.stderr))


class TencentCloudClient:
    """Base class for Tencent Cloud service clients with async support."""

    def __init__(
        self,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint: str = "",
        timeout: int = 60,
        max_retries: int = 3
    ):
        self.secret_id = secret_id or settings.tencent_secret_id
        self.secret_key = secret_key or settings.tencent_secret_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self._credential = None
        self._client_profile = None

    def _get_credential(self) -> credential.Credential:
        """Get or create credential."""
        if self._credential is None:
            self._credential = credential.Credential(
                self.secret_id, self.secret_key
            )
        return self._credential

    def _get_client_profile(self) -> ClientProfile:
        """Get or create client profile with retry and timeout settings."""
        if self._client_profile is None:
            http_profile = HttpProfile()
            http_profile.endpoint = self.endpoint
            http_profile.keepAlive = True
            http_profile.reqTimeout = self.timeout

            self._client_profile = ClientProfile()
            self._client_profile.httpProfile = http_profile
            # Add retry mechanism for network/rate limit errors
            self._client_profile.retryer = retry.StandardRetryer(
                max_attempts=self.max_retries,
                logger=retry_logger
            )
        return self._client_profile
