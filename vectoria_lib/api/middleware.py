import collections
import logging
import time
import uuid

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request with method, path, status code and duration."""

    async def dispatch(self, request, call_next):
        request_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        logger.info(
            "[%s] %s %s → %d (%.3fs)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter per client IP.

    Rules (POST only):
      /v1/chat/completions  — 10 req / 60 s  (LLM inference)
      /documents/ingest     —  5 req / 60 s  (indexing, CPU/GPU intensive)
      /documents/cancel     — 10 req / 60 s  (lightweight but avoid abuse)
    """

    # (limit, window_seconds) per POST path
    _RULES: dict[str, tuple[int, int]] = {
        "/v1/chat/completions": (10, 60),
        "/documents/ingest": (5, 60),
        "/documents/cancel": (10, 60),
    }

    def __init__(self, app):
        super().__init__(app)
        # (ip, path) → deque of request timestamps
        self._requests: dict[tuple[str, str], collections.deque] = collections.defaultdict(collections.deque)

    async def dispatch(self, request, call_next):
        path = request.url.path
        if request.method == "POST" and path in self._RULES:
            limit, window_secs = self._RULES[path]
            ip = request.client.host if request.client else "unknown"
            now = time.monotonic()
            key = (ip, path)
            window = self._requests[key]

            while window and window[0] < now - window_secs:
                window.popleft()

            if len(window) >= limit:
                retry_after = int(window_secs - (now - window[0])) + 1
                logger.warning("Rate limit exceeded — IP=%s path=%s", ip, path)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": "Too many requests",
                            "type": "rate_limit_error",
                            "code": "rate_limit_exceeded",
                        }
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            window.append(now)

        return await call_next(request)
