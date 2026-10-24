"""Application API keys: only a SHA-256 of the key is stored."""

from __future__ import annotations

import hashlib
import secrets

PREFIX = "ppag_"


def new_key() -> str:
    return PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256((key or "").encode()).hexdigest()
