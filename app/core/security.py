"""
Input checks and API-key auth
=============================
`sanitize_query` is a cheap first filter for user text sent to LLM chains: it enforces the
length limit and rejects a few well-known prompt-injection phrases. It is easy to bypass, so
it is not the real defense; prompts keep user text in clearly delimited sections and tell the
model to treat it as data.

`require_api_key` protects write endpoints with the `X-API-Key` header.
"""
import hmac
import re

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

# Patterns commonly used in prompt-injection attacks
_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions"
    r"|system\s+prompt"
    r"|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>"
    r"|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)",
    re.IGNORECASE,
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def sanitize_query(text: str) -> str:
    """
    Check user-supplied text before passing it to LLM chains.

    Raises HTTPException(400) if the input exceeds the maximum length or
    contains known prompt-injection patterns.

    Returns the stripped input on success.
    """
    max_length = get_settings().MAX_QUERY_LENGTH
    if len(text) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"Input exceeds maximum length of {max_length} characters.",
        )
    if _INJECTION_RE.search(text):
        raise HTTPException(
            status_code=400,
            detail="Input contains disallowed content.",
        )
    return text.strip()


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """
    Reject the request unless X-API-Key matches settings.API_KEY.

    When API_KEY is not configured (local development), every request is allowed;
    app startup logs a warning in that case.
    """
    expected = get_settings().API_KEY
    if not expected:
        return
    if not api_key or not hmac.compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
