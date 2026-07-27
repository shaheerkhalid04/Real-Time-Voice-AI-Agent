"""Text-to-speech through ElevenLabs.

`eleven_flash_v2_5` is the low-latency model, which matters a lot when the
reply is meant to feel like a conversation rather than a download.
"""

from __future__ import annotations

import httpx

from .config import ELEVENLABS_BASE_URL, Settings, get_settings

# mp3 for the browser (it decodes it natively); signed 16-bit PCM at 24 kHz for
# the CLI, which pipes straight into the sound card without an mp3 decoder.
FORMAT_MP3 = "mp3_44100_128"
FORMAT_PCM = "pcm_24000"

MAX_CHARS = 2500


class TTSError(RuntimeError):
    """Speech synthesis failed."""


async def synthesize(
    text: str,
    *,
    voice_id: str | None = None,
    output_format: str = FORMAT_MP3,
    stability: float = 0.45,
    similarity_boost: float = 0.75,
    speed: float = 1.0,
    settings: Settings | None = None,
) -> bytes:
    """Render `text` as audio bytes in `output_format`."""
    settings = settings or get_settings()
    api_key = settings.require_elevenlabs()

    text = (text or "").strip()
    if not text:
        raise TTSError("There is nothing to speak.")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0] + "..."

    voice = voice_id or settings.voice_id
    payload = {
        "text": text,
        "model_id": settings.tts_model,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "speed": speed,
            "use_speaker_boost": True,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            response = await client.post(
                f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice}",
                params={"output_format": output_format},
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.RequestError as exc:
        raise TTSError(f"Could not reach the speech service: {exc}") from exc

    if response.status_code != 200:
        raise TTSError(_describe_error(response))

    audio = response.content
    if not audio:
        raise TTSError("The speech service returned an empty clip.")
    return audio


async def list_voices(settings: Settings | None = None) -> list[dict]:
    """Return the voices available on the configured account.

    Returns an empty list rather than raising: the voice picker is a nicety,
    and a failure there should never break a conversation.
    """
    settings = settings or get_settings()
    if not settings.has_tts:
        return []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{ELEVENLABS_BASE_URL}/voices",
                headers={"xi-api-key": settings.elevenlabs_api_key},
            )
        if response.status_code != 200:
            return []
        voices = response.json().get("voices", [])
    except Exception:
        return []

    return [
        {
            "id": voice.get("voice_id"),
            "name": voice.get("name"),
            "labels": voice.get("labels", {}),
            "preview": voice.get("preview_url"),
        }
        for voice in voices
        if voice.get("voice_id")
    ]


def _describe_error(response: httpx.Response) -> str:
    if response.status_code == 401:
        return "ElevenLabs rejected the API key. Check ELEVENLABS_API_KEY."
    if response.status_code == 429:
        return "ElevenLabs rate limit or quota reached."
    try:
        detail = response.json().get("detail")
        if isinstance(detail, dict) and detail.get("message"):
            return f"Speech synthesis failed: {detail['message']}"
        if isinstance(detail, str):
            return f"Speech synthesis failed: {detail}"
    except Exception:
        pass
    return f"Speech synthesis failed with HTTP {response.status_code}."
