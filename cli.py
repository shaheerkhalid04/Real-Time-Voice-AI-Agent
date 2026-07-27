"""Terminal voice agent — press Enter, talk, stop talking, hear the answer.

    python cli.py                 hold a conversation
    python cli.py --list-devices  show input devices
    python cli.py --mute          print replies instead of speaking them

Recording ends by itself once you go quiet, so a turn is one keypress.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import time
import wave
from typing import Any

from voice_agent import asr, llm, tts
from voice_agent.config import ConfigError, get_settings

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_MS = 30
SILENCE_HANGOVER_S = 1.2
MAX_TURN_S = 30.0
CALIBRATION_S = 0.4


# --------------------------------------------------------------------------
# Console output — rich when available, plain text otherwise.
# --------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel

    _console = Console()

    def say(markup: str = "") -> None:
        _console.print(markup)

    def panel(body: str, title: str, colour: str) -> None:
        _console.print(Panel(body, title=title, border_style=colour, padding=(0, 1)))

except ImportError:  # pragma: no cover - cosmetic fallback
    _console = None

    def say(markup: str = "") -> None:
        import re

        print(re.sub(r"\[/?[^\]]+\]", "", markup))

    def panel(body: str, title: str, colour: str) -> None:
        print(f"\n--- {title} ---\n{body}\n")


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

def record_until_silence(device: int | None = None) -> bytes:
    """Capture from the microphone until the speaker pauses. Returns WAV bytes."""
    import numpy as np
    import sounddevice as sd

    block = int(SAMPLE_RATE * BLOCK_MS / 1000)
    frames: list[Any] = []
    noise_floor = 0.008
    speech_seen = False
    silence_started: float | None = None
    started = time.perf_counter()

    say("[bold red]● recording[/bold red] [dim]— stop talking when you're done[/dim]")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=block,
        device=device,
    ) as stream:
        while True:
            chunk, overflowed = stream.read(block)
            if overflowed:
                continue
            frames.append(chunk.copy())

            samples = chunk.astype(np.float32) / 32768.0
            level = float(np.sqrt(np.mean(samples**2)))
            elapsed = time.perf_counter() - started

            if elapsed < CALIBRATION_S:
                noise_floor = max(noise_floor, level)
                continue

            threshold = max(0.014, noise_floor * 2.4)
            if level > threshold:
                speech_seen = True
                silence_started = None
            elif speech_seen:
                silence_started = silence_started or time.perf_counter()
                if time.perf_counter() - silence_started > SILENCE_HANGOVER_S:
                    break

            if elapsed > MAX_TURN_S:
                say("[yellow]Reached the 30 second limit.[/yellow]")
                break

    audio = np.concatenate(frames) if frames else np.zeros((0, CHANNELS), dtype="int16")
    return _to_wav(audio.tobytes())


def _to_wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)
    return buffer.getvalue()


def play_pcm(pcm: bytes, sample_rate: int = 24000) -> None:
    import numpy as np
    import sounddevice as sd

    audio = np.frombuffer(pcm, dtype=np.int16)
    sd.play(audio, sample_rate)
    try:
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()


# --------------------------------------------------------------------------
# Conversation loop
# --------------------------------------------------------------------------

async def converse(args: argparse.Namespace) -> int:
    settings = get_settings()

    if not settings.has_asr:
        say("[bold red]GROQ_API_KEY is not set.[/bold red] Copy .env.example to .env and add your key.")
        return 1

    speak_replies = not args.mute and settings.has_tts
    if not args.mute and not settings.has_tts:
        say("[yellow]No ELEVENLABS_API_KEY — replies will be printed, not spoken.[/yellow]")

    say()
    say("[bold]Real-Time Voice AI Agent[/bold]")
    say(f"[dim]{settings.asr_model} → {settings.llm_model}"
        f"{' → ' + settings.tts_model if speak_replies else ''}[/dim]")
    say("[dim]Enter to talk · 'q' then Enter to quit[/dim]")
    say()

    history: list[dict[str, Any]] = []

    while True:
        try:
            command = input("\033[90m[Enter to talk]\033[0m ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            say("\n[dim]Bye.[/dim]")
            return 0
        if command in {"q", "quit", "exit"}:
            say("[dim]Bye.[/dim]")
            return 0

        try:
            wav = record_until_silence(args.device)
        except Exception as exc:
            say(f"[bold red]Microphone error:[/bold red] {exc}")
            say("[dim]Try 'python cli.py --list-devices' and pass --device N.[/dim]")
            return 1

        if len(wav) < 8000:
            say("[yellow]That was too short — try again.[/yellow]\n")
            continue

        started = time.perf_counter()
        try:
            text = await asr.transcribe(
                wav, filename="speech.wav", content_type="audio/wav", settings=settings
            )
        except (asr.ASRError, ConfigError) as exc:
            say(f"[bold red]Transcription failed:[/bold red] {exc}\n")
            continue
        asr_ms = int((time.perf_counter() - started) * 1000)

        if not text:
            say("[yellow]I did not catch anything.[/yellow]\n")
            continue

        panel(text, f"You · {asr_ms} ms", "blue")

        history.append({"role": "user", "content": text})
        started = time.perf_counter()
        try:
            reply = await llm.respond(history, settings=settings)
        except (llm.LLMError, ConfigError) as exc:
            say(f"[bold red]The agent failed:[/bold red] {exc}\n")
            history.pop()
            continue
        llm_ms = int((time.perf_counter() - started) * 1000)

        history = [m for m in reply.messages if m.get("role") != "system"][-24:]

        for tool in reply.tools:
            say(f"  [magenta]⚙ {tool.name}[/magenta][dim]({_short(tool.arguments)}) "
                f"→ {_short(tool.result, 90)}[/dim]")

        panel(reply.text, f"Agent · {llm_ms} ms", "green")

        if speak_replies:
            started = time.perf_counter()
            try:
                audio = await tts.synthesize(
                    reply.text, output_format=tts.FORMAT_PCM, settings=settings
                )
                say(f"[dim]  speaking… ({int((time.perf_counter() - started) * 1000)} ms)[/dim]")
                play_pcm(audio)
            except (tts.TTSError, ConfigError) as exc:
                say(f"[yellow]Could not speak the reply:[/yellow] {exc}")

        say()


def _short(value: Any, limit: int = 60) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def list_devices() -> int:
    try:
        import sounddevice as sd
    except ImportError:
        say("[bold red]sounddevice is not installed.[/bold red] "
            "Run: pip install -r requirements-cli.txt")
        return 1

    say("[bold]Input devices[/bold]")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            say(f"  [cyan]{index:>2}[/cyan]  {device['name']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Talk to the voice agent from your terminal.")
    parser.add_argument("--device", type=int, default=None, help="input device index")
    parser.add_argument("--mute", action="store_true", help="print replies instead of speaking")
    parser.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        return list_devices()

    try:
        import numpy  # noqa: F401
        import sounddevice  # noqa: F401
    except ImportError:
        say("[bold red]Missing audio dependencies.[/bold red] "
            "Run: pip install -r requirements.txt -r requirements-cli.txt")
        return 1

    try:
        return asyncio.run(converse(args))
    except KeyboardInterrupt:
        say("\n[dim]Bye.[/dim]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
