"""Real-Time Voice AI Agent — core package.

Three stages wired together:
    audio in  ->  asr.transcribe()  ->  llm.respond()  ->  tts.synthesize()  ->  audio out
"""

from .config import Settings, get_settings
from .asr import transcribe
from .llm import respond
from .tts import synthesize, list_voices
from .tools import TOOL_SCHEMAS, run_tool

__version__ = "1.0.0"

__all__ = [
    "Settings",
    "get_settings",
    "transcribe",
    "respond",
    "synthesize",
    "list_voices",
    "TOOL_SCHEMAS",
    "run_tool",
]
