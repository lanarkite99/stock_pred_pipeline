import os
import time
from collections import defaultdict
from threading import Lock

import backend.state as app_state
from fastapi import Request
from fastapi.responses import JSONResponse

_WINDOW_SECONDS = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
_MAX_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60")))
_EXEMPT_PREFIXES = ("/", "/health", "/metrics", "/docs", "/redoc", "/openapi.json")
_local_counts = defaultdict(int)
_local_lock = Lock()


def _is_exempt(path: str) -> bool:
    return path in _EXEMPT_PREFIXES or path.startswith("/docs") or path.startswith("/openapi")


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _bucket(now: float) -> int:
    return int(now // _WINDOW_SECONDS)


def _cleanup_local_counts(current_bucket: int) -> None:
    if len(_local_counts) < 5000:
        return
    stale = [key for key in _local_counts if key[1] < current_bucket - 1]
    for key in stale:
        _local_counts.pop(key, None)


def check_rate_limit(request: Request):
    path = request.url.path
    if _is_exempt(path):
        return True, {}

    now = time.time()
    bucket = _bucket(now)
    ip = _client_ip(request)
    remaining = _MAX_REQUESTS
    count = 0

    if app_state.redis_client is not None:
        key = f"rate_limit:{ip}:{bucket}"
        try:
            count = int(app_state.redis_client.incr(key))
            if count == 1:
                app_state.redis_client.expire(key, _WINDOW_SECONDS + 1)
        except Exception:
            count = 0

    if count == 0:
        with _local_lock:
            local_key = (ip, bucket)
            _local_counts[local_key] += 1
            count = _local_counts[local_key]
            _cleanup_local_counts(bucket)

    remaining = max(0, _MAX_REQUESTS - count)
    reset_at = (bucket + 1) * _WINDOW_SECONDS
    headers = {
        "X-RateLimit-Limit": str(_MAX_REQUESTS),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset_at),
    }
    return count <= _MAX_REQUESTS, headers


def rate_limit_response(headers):
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"rate limit exceeded: max {_MAX_REQUESTS} requests per {_WINDOW_SECONDS} seconds",
        },
        headers=headers,
    )
