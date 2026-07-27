"""Configuration read from environment variables.

Nothing here throws on import — a missing key only becomes an error when the
feature that needs it is actually used, so the app can boot (and the web UI can
fall back to browser speech) with a partial setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

# Loading a .env file is a local-development convenience. On Vercel the
# variables come from the project settings, and python-dotenv is not installed.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# "Rachel" from the ElevenLabs public voice library. Free accounts cannot use
# library voices over the API, so this is only a last resort — tts.py replaces
# it with the first voice the account actually owns.
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

DEFAULT_SYSTEM_PROMPT = (
    "You are a real-time voice assistant. Your replies are read aloud, so keep "
    "them short, natural and conversational — usually one to three sentences. "
    "Never use markdown, bullet points, emoji or code blocks, because none of "
    "that can be spoken. Write numbers, dates and units the way a person would "
    "say them. If a tool can answer the question, call it instead of guessing. "
    "If you did not understand the audio, say so plainly and ask for a repeat."
)


class ConfigError(RuntimeError):
    """Raised when a request needs an API key that was never configured."""


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = ""
    elevenlabs_api_key: str = ""

    asr_model: str = "whisper-large-v3-turbo"
    llm_model: str = "llama-3.3-70b-versatile"
    tts_model: str = "eleven_flash_v2_5"

    voice_id: str = DEFAULT_VOICE_ID
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    temperature: float = 0.6
    max_tokens: int = 400
    request_timeout: float = 45.0

    # Tool calls the agent may chain in a single turn before we force a
    # plain-text answer. Keeps a confused model from looping forever.
    max_tool_rounds: int = 3

    allowed_origins: list[str] = field(default_factory=lambda: ["*"])

    @property
    def has_asr(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_llm(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_tts(self) -> bool:
        return bool(self.elevenlabs_api_key)

    def require_groq(self) -> str:
        if not self.groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Add it to your .env file or to the "
                "environment variables of your deployment."
            )
        return self.groq_api_key

    def require_elevenlabs(self) -> str:
        if not self.elevenlabs_api_key:
            raise ConfigError(
                "ELEVENLABS_API_KEY is not set. Add it to your .env file, or "
                "turn on browser speech in the settings panel."
            )
        return self.elevenlabs_api_key


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _float_env(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = _env("ALLOWED_ORIGINS", "*")
    return Settings(
        groq_api_key=_env("GROQ_API_KEY"),
        elevenlabs_api_key=_env("ELEVENLABS_API_KEY"),
        asr_model=_env("ASR_MODEL", "whisper-large-v3-turbo"),
        llm_model=_env("LLM_MODEL", "llama-3.3-70b-versatile"),
        tts_model=_env("TTS_MODEL", "eleven_flash_v2_5"),
        voice_id=_env("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID),
        system_prompt=_env("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        temperature=_float_env("LLM_TEMPERATURE", 0.6),
        max_tokens=_int_env("LLM_MAX_TOKENS", 400),
        request_timeout=_float_env("REQUEST_TIMEOUT", 45.0),
        allowed_origins=[o.strip() for o in origins.split(",") if o.strip()],
    )
