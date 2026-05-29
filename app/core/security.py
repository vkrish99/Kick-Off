"""
Input Sanitization
==================
Validate and sanitize user-supplied text before passing it to LLM chains.
Protects against prompt-injection attacks and oversized inputs.
"""
import re
from fastapi import HTTPException

# Patterns commonly used in prompt-injection attacks
_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions"
    r"|system\s+prompt"
    r"|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>"
    r"|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)",
    re.IGNORECASE,
)

MAX_INPUT_LENGTH = 2000


def sanitize_query(text: str) -> str:
    """
    Sanitize user-supplied text before passing to LLM chains.

    Raises HTTPException(400) if the input exceeds the maximum length or
    contains known prompt-injection patterns.

    Returns the stripped input on success.
    """
    if len(text) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Input exceeds maximum length of {MAX_INPUT_LENGTH} characters.",
        )
    if _INJECTION_RE.search(text):
        raise HTTPException(
            status_code=400,
            detail="Input contains disallowed content.",
        )
    return text.strip()
