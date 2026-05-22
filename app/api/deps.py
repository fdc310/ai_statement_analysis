"""
Shared FastAPI dependencies.
"""
from typing import Optional

from fastapi import Header, HTTPException

from app.core.config import settings
from app.core.security import aes_service


def verify_signature_header(
    x_signature: Optional[str] = Header(None, alias="X-Signature"),
) -> None:
    """Verify AES signature from the X-Signature header."""
    if not x_signature:
        raise HTTPException(status_code=401, detail="Missing X-Signature header")

    success, message = aes_service.verify_signature(
        x_signature,
        max_age_seconds=settings.request_expire_seconds,
    )
    if not success:
        raise HTTPException(status_code=401, detail=message)
