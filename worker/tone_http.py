import re
from typing import Any
from urllib.parse import urlsplit

try:
    from .toning import tone_request
except ImportError:  # Cloudflare loads modules beside main.py.
    from toning import tone_request

SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
MAX_REQUEST_BYTES = 4096


def response_headers() -> dict[str, str]:
    return {
        "cache-control": "no-store",
        "content-type": "application/json; charset=utf-8",
        "x-content-type-options": "nosniff",
    }


def stream_response_headers() -> dict[str, str]:
    return {
        "cache-control": "no-store",
        "content-type": "application/x-ndjson; charset=utf-8",
        "x-accel-buffering": "no",
        "x-content-type-options": "nosniff",
    }


def wants_stream(request: Any) -> bool:
    accept = request.headers.get("accept")
    return isinstance(accept, str) and "application/x-ndjson" in accept.lower()


def _same_origin(origin: str | None, request_url: str) -> bool:
    if origin is None:
        return True
    try:
        supplied = urlsplit(origin)
        requested = urlsplit(request_url)
        supplied_port = supplied.port or (443 if supplied.scheme == "https" else 80)
        requested_port = requested.port or (443 if requested.scheme == "https" else 80)
        return (
            supplied.scheme in {"http", "https"}
            and supplied.hostname is not None
            and supplied.scheme == requested.scheme
            and supplied.hostname.lower() == (requested.hostname or "").lower()
            and supplied_port == requested_port
        )
    except ValueError:
        return False


def _limit_succeeded(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("success") is True
    return getattr(value, "success", False) is True


async def api_dispatch(
    request: Any, env: Any, emit=None
) -> tuple[dict[str, Any], int] | None:
    if urlsplit(request.url).path != "/api/tone":
        return None
    if request.method != "POST":
        return {"error": "Method not allowed"}, 405

    content_length = request.headers.get("content-length")
    try:
        if content_length is not None and int(content_length) > MAX_REQUEST_BYTES:
            return {"error": "Request too large"}, 413
    except (TypeError, ValueError):
        return {"error": "Invalid content length"}, 400

    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}, 400

    session_id = request.headers.get("x-tone-session")
    if not isinstance(session_id, str) or not SESSION_PATTERN.fullmatch(session_id):
        return {"error": "Invalid session"}, 400

    origin = request.headers.get("origin")
    if not _same_origin(origin, request.url):
        return {"error": "Cross-origin requests are not allowed"}, 403

    session_result = await env.SESSION_LIMITER.limit({"key": session_id})
    if not _limit_succeeded(session_result):
        return {"error": "Easy, tiger. Try again soon."}, 429

    global_result = await env.GLOBAL_LIMITER.limit({"key": "tone-jev"})
    if not _limit_succeeded(global_result):
        return {"error": "The tone queue is full. Try again soon."}, 429

    async def run_ai(model: str, value: dict[str, Any]) -> Any:
        return await env.AI.run(model, value)

    return await tone_request(body, request.url, origin, run_ai, emit)
