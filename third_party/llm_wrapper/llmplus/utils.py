import hashlib
import logging
from json import JSONDecodeError

import httpx
import orjson
from openai import APIStatusError
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def stable_hash(obj: object) -> str:
    """md5 of a json‑serialisable object"""
    # dumped = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    dumped = orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)
    return hashlib.md5(dumped).hexdigest()


def transient_retry(*, attempts: int = 5, wait_min: int = 1, wait_max: int = 120):
    def _is_transient(exc: Exception):
        # classify by type or status, not by substring
        if isinstance(exc, JSONDecodeError):
            return True
        if isinstance(
            exc,
            (
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.RemoteProtocolError,
                httpx.ConnectError,
            ),
        ):
            return True
        if isinstance(exc, APIStatusError):
            # Retry typical transient codes
            try:
                status = exc.status_code
            except Exception:
                status = None
            # 429: too many requests
            # 500-599: server errors
            return status == 429 or (status is not None and 500 <= status < 600)
        # final fallback on message substrings (optional)
        s = str(exc)
        return any(
            k in s for k in ("Rate limit", "Bad gateway", "temporarily unavailable")
        )

    return retry(
        wait=wait_exponential(min=wait_min, max=wait_max),
        stop=stop_after_attempt(attempts),
        retry=retry_if_exception_type(Exception) & retry_if_exception(_is_transient),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
        reraise=True,
    )
