"""Speech-to-text through Groq's hosted Whisper endpoint.

Groq exposes an OpenAI-compatible audio API, so this is a single multipart
POST. `whisper-large-v3-turbo` is the fast model; swap it with ASR_MODEL.
"""

from __future__ import annotations

import httpx

from .config import GROQ_BASE_URL, Settings, get_settings

# Extensions the endpoint accepts. The browser records webm/opus, the CLI
# uploads wav, and both are on the list.
SUPPORTED_EXTENSIONS = (
    "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "opus", "wav", "webm",
)

MAX_AUDIO_BYTES = 24 * 1024 * 1024


class ASRError(RuntimeError):
    """Transcription failed."""


async def transcribe(
    audio: bytes,
    *,
    filename: str = "speech.webm",
    content_type: str = "audio/webm",
    language: str | None = None,
    prompt: str | None = None,
    settings: Settings | None = None,
) -> str:
    """Return the text spoken in `audio`, or an empty string for silence."""
    settings = settings or get_settings()
    api_key = settings.require_groq()

    if not audio:
        raise ASRError("No audio was received.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ASRError("That clip is too long. Keep recordings under 25 MB.")

    data: dict[str, str] = {
        "model": settings.asr_model,
        "response_format": "json",
        # Nudges Whisper away from hallucinating text into background noise.
        "temperature": "0",
    }
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            response = await client.post(
                f"{GROQ_BASE_URL}/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (filename, audio, content_type)},
                data=data,
            )
    except httpx.RequestError as exc:
        raise ASRError(f"Could not reach the transcription service: {exc}") from exc

    if response.status_code != 200:
        raise ASRError(_describe_error(response))

    text = (response.json().get("text") or "").strip()
    return "" if _is_noise(text) else text


# Whisper tends to emit one of these when handed near-silence.
_NOISE_TRANSCRIPTS = {
    "you", "thank you.", "thanks for watching!", "bye.", ".", "...", "[blank_audio]",
    "thank you for watching!", "please subscribe.", "subtitles by the amara.org community",
}


def _is_noise(text: str) -> bool:
    stripped = text.strip().lower()
    return not stripped or stripped in _NOISE_TRANSCRIPTS


def _describe_error(response: httpx.Response) -> str:
    if response.status_code == 401:
        return "Groq rejected the API key. Check GROQ_API_KEY."
    if response.status_code == 429:
        return "Groq rate limit reached. Wait a moment and try again."
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message")
        if message:
            return f"Transcription failed: {message}"
    except Exception:
        pass
    return f"Transcription failed with HTTP {response.status_code}."
