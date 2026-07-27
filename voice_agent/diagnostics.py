"""Key diagnostics for /api/health.

The plain health check only reports whether a key is *present*, which is not
the same as it working — a revoked or mistyped key looks configured and then
fails on the first real request. These checks close that gap without ever
putting any part of a secret in a response.
"""

from __future__ import annotations

from typing import Any

import httpx

from .config import ELEVENLABS_BASE_URL, GROQ_BASE_URL, Settings

# Groq keys are issued as gsk_..., ElevenLabs keys as sk_.... A key in the
# wrong variable is the single most common setup mistake, and comparing
# prefixes catches it without revealing a character of either value.
GROQ_PREFIX = "gsk_"
ELEVENLABS_PREFIX = "sk_"


def key_shapes(settings: Settings) -> dict[str, Any]:
    """Report whether each key looks like the kind of key it should be."""
    groq, eleven = settings.groq_api_key, settings.elevenlabs_api_key
    return {
        "groq": _shape(groq, GROQ_PREFIX, ELEVENLABS_PREFIX),
        "elevenlabs": _shape(eleven, ELEVENLABS_PREFIX, GROQ_PREFIX),
    }


def _shape(key: str, expected: str, other: str) -> dict[str, Any]:
    if not key:
        return {"present": False, "looks_right": None, "note": "not set"}

    if key.startswith(expected):
        # gsk_ also starts with g, so check the more specific prefix first.
        return {"present": True, "looks_right": True, "note": "ok"}

    if key.startswith(other):
        return {
            "present": True,
            "looks_right": False,
            "note": f"starts with {other} — this looks like the other provider's key",
        }

    return {
        "present": True,
        "looks_right": False,
        "note": f"does not start with {expected} — check for a stray quote, space or truncation",
    }


async def verify_keys(settings: Settings) -> dict[str, str]:
    """Ask each provider whether it actually accepts its key."""
    return {
        "groq": await _verify(
            f"{GROQ_BASE_URL}/models",
            {"Authorization": f"Bearer {settings.groq_api_key}"},
            settings.has_asr,
        ),
        # /voices rather than /user: it is the call the app itself makes, and
        # a key scoped to speech only is refused by /user while working fine.
        "elevenlabs": await _verify(
            f"{ELEVENLABS_BASE_URL}/voices",
            {"xi-api-key": settings.elevenlabs_api_key},
            settings.has_tts,
        ),
    }


async def _verify(url: str, headers: dict[str, str], configured: bool) -> str:
    if not configured:
        return "not configured"
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(url, headers=headers)
    except httpx.RequestError:
        return "unreachable"

    if response.status_code == 200:
        return "accepted"
    if response.status_code in (401, 403):
        return "rejected — the key is wrong, revoked, or for a different account"
    if response.status_code == 429:
        return "accepted but rate limited"
    return f"unexpected HTTP {response.status_code}"
