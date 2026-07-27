"""HTTP API for the voice agent — runs locally with uvicorn and on Vercel.

Vercel's Python runtime picks up the module-level `app` and serves it as an
ASGI function; the same object is what `uvicorn api.index:app` runs in
development, so there is only ever one code path to debug.

Routes are registered twice, bare and under /api, because the deployed app is
reached through a rewrite while local development hits the app directly.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path
from typing import Any

# Import the sibling package when this file is executed as a standalone
# serverless entry point rather than as part of an installed package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from voice_agent import asr, diagnostics, llm, tts
from voice_agent.config import ConfigError, get_settings

app = FastAPI(
    title="Real-Time Voice AI Agent",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

router = APIRouter()


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    use_tools: bool = True
    system_prompt: str | None = None


class SpeakRequest(BaseModel):
    text: str
    voice_id: str | None = None
    speed: float = Field(default=1.0, ge=0.7, le=1.2)


class ConverseRequest(BaseModel):
    """One round trip: transcript in, reply text and spoken audio out."""

    messages: list[Message] = Field(default_factory=list)
    use_tools: bool = True
    system_prompt: str | None = None
    voice_id: str | None = None
    speak: bool = True


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@router.get("/health")
async def health(verify: bool = False) -> dict[str, Any]:
    """What the server can actually do — the UI adapts its controls to this.

    `?verify=1` additionally asks each provider whether it accepts its key,
    which is the difference between "a key is set" and "a key works".
    """
    payload: dict[str, Any] = {
        "status": "ok",
        "asr": settings.has_asr,
        "llm": settings.has_llm,
        "tts": settings.has_tts,
        "models": {
            "asr": settings.asr_model,
            "llm": settings.llm_model,
            "tts": settings.tts_model,
        },
        "default_voice": settings.voice_id,
        "tools": [schema["function"]["name"] for schema in llm.TOOL_SCHEMAS],
        "keys": diagnostics.key_shapes(settings),
    }
    if verify:
        payload["verified"] = await diagnostics.verify_keys(settings)
    return payload


@router.get("/voices")
async def voices() -> dict[str, Any]:
    return {"voices": await tts.list_voices(settings)}


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> JSONResponse:
    started = time.perf_counter()
    payload = await audio.read()

    try:
        text = await asr.transcribe(
            payload,
            filename=audio.filename or "speech.webm",
            content_type=audio.content_type or "audio/webm",
            language=language or None,
            settings=settings,
        )
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except asr.ASRError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return JSONResponse(
        {
            "text": text,
            "empty": not text,
            "ms": _elapsed_ms(started),
            "bytes": len(payload),
        }
    )


@router.post("/chat")
async def chat(request: ChatRequest) -> JSONResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages were sent.")

    started = time.perf_counter()
    try:
        reply = await llm.respond(
            [_clean(message) for message in request.messages],
            use_tools=request.use_tools,
            system_prompt=request.system_prompt,
            settings=settings,
        )
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return JSONResponse({**reply.to_dict(), "ms": _elapsed_ms(started)})


@router.post("/speak")
async def speak(request: SpeakRequest) -> Response:
    started = time.perf_counter()
    try:
        audio = await tts.synthesize(
            request.text,
            voice_id=request.voice_id,
            speed=request.speed,
            settings=settings,
        )
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except tts.TTSError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "X-Latency-Ms": str(_elapsed_ms(started)),
            "Cache-Control": "no-store",
        },
    )


@router.post("/converse")
async def converse(request: ConverseRequest) -> JSONResponse:
    """Think and speak in a single request, for clients that prefer one call."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages were sent.")

    started = time.perf_counter()
    try:
        reply = await llm.respond(
            [_clean(message) for message in request.messages],
            use_tools=request.use_tools,
            system_prompt=request.system_prompt,
            settings=settings,
        )
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    llm_ms = _elapsed_ms(started)
    audio_b64: str | None = None
    tts_ms = 0
    tts_error: str | None = None

    if request.speak and settings.has_tts:
        tts_started = time.perf_counter()
        try:
            audio = await tts.synthesize(
                reply.text, voice_id=request.voice_id, settings=settings
            )
            audio_b64 = base64.b64encode(audio).decode("ascii")
        except (ConfigError, tts.TTSError) as exc:
            # A voice failure should not lose the answer — the client can fall
            # back to browser speech and still show the text.
            tts_error = str(exc)
        tts_ms = _elapsed_ms(tts_started)

    return JSONResponse(
        {
            **reply.to_dict(),
            "audio": audio_b64,
            "audio_format": "audio/mpeg" if audio_b64 else None,
            "tts_error": tts_error,
            "ms": {"llm": llm_ms, "tts": tts_ms},
        }
    )


def _clean(message: Message) -> dict[str, Any]:
    return message.model_dump(exclude_none=True)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


app.include_router(router)
app.include_router(router, prefix="/api")

# In development this one process serves the UI as well, so `uvicorn
# api.index:app` is the whole app. In production Vercel's CDN serves public/
# and this function only ever sees /api/*.
PUBLIC_DIR = ROOT / "public"

if PUBLIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="ui")
else:

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        # The UI is on the CDN rather than in this bundle, so hand it over.
        return RedirectResponse("/index.html")
