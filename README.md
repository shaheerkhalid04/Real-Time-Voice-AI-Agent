# Real-Time Voice AI Agent

Talk to it, and it talks back. Your speech is transcribed, answered by a
language model that can call tools when it needs real information, and read
back to you in a natural voice.

**Live app → https://real-time-voice-ai-agent-two.vercel.app**

There are two ways to use it: a browser app with a live waveform, silence
detection and a hands-free mode, and a terminal agent for when you would
rather stay in the shell.

---

## What it does

```
    speech  ──▶  Whisper (Groq)  ──▶  Llama 3.3 (Groq)  ──▶  ElevenLabs  ──▶  speech
                  transcribe          think + use tools        synthesize
```

- **Speech-to-text** — `whisper-large-v3-turbo` on Groq, reached over the
  OpenAI-compatible audio API.
- **The agent** — `llama-3.3-70b-versatile` on Groq, with a tool-calling loop
  that resolves up to three rounds of calls before it has to answer in words.
- **Text-to-speech** — ElevenLabs `eleven_flash_v2_5`, the low-latency model,
  because a reply that arrives late does not feel like a conversation.

Every stage is timed and the timings are shown in the UI, so it is obvious
which part of the pipeline is slow on any given turn.

## Features

**Browser app**

- Hold-to-talk with <kbd>Space</kbd>, or click the microphone to toggle.
- **Automatic turn-taking.** The recorder measures the room's noise floor for
  the first 400 ms, then ends your turn once you have been quiet for a beat.
  Adjustable from 0.4 s to 3 s.
- **Hands-free mode** — it starts listening again after every reply, so you can
  hold a whole conversation without touching anything.
- Live waveform driven by the Web Audio analyser, showing your voice while you
  speak and the agent's while it answers.
- A pipeline readout — Listen, Transcribe, Think, Speak — that lights up stage
  by stage with the milliseconds each one took.
- Tool calls appear as chips in the transcript; click one to see the exact
  arguments and the raw result.
- Barge-in: start talking and playback stops.
- Type instead of talking, replay any answer, export the transcript, switch
  between the light and dark theme, pick a voice and set the speaking rate.
- Degrades honestly. No ElevenLabs key falls back to the browser's own voice;
  no microphone permission leaves typing working and says so.

**Terminal agent**

- One keypress per turn, silence-detected recording, tool calls printed inline.
- Plays PCM straight to the sound card, so there is no mp3 decoder to install.

## Quick start

You need **Python 3.10+** and a free [Groq API key](https://console.groq.com/keys).
An [ElevenLabs key](https://elevenlabs.io/app/settings/api-keys) is optional —
without one the browser speaks the replies itself.

```bash
git clone https://github.com/shaheerkhalid04/Real-Time-Voice-AI-Agent.git
cd Real-Time-Voice-AI-Agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env             # then put your keys in it
```

**Run the web app:**

```bash
uvicorn api.index:app --reload --port 8000
```

Open <http://localhost:8000>. In development this one process serves both the
API and the UI.

**Run the terminal agent:**

```bash
pip install -r requirements-cli.txt
python cli.py
```

Press Enter, say something, and stop talking — it takes it from there. Use
`python cli.py --list-devices` if it picks the wrong microphone, and
`--mute` to read the replies instead of hearing them.

> The microphone only works over HTTPS or on `localhost`. That is a browser
> rule, not an app setting.

## Configuration

Everything is read from the environment. Only the first one is required.

| Variable | Default | What it does |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Required. Powers both transcription and the agent. |
| `ELEVENLABS_API_KEY` | — | Optional. Without it, replies use the browser voice. |
| `ELEVENLABS_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | Which voice answers. |
| `ASR_MODEL` | `whisper-large-v3-turbo` | Speech-to-text model. |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Agent model. |
| `TTS_MODEL` | `eleven_flash_v2_5` | Speech model. |
| `LLM_TEMPERATURE` | `0.6` | Higher is more varied. |
| `LLM_MAX_TOKENS` | `400` | Spoken answers should be short. |
| `REQUEST_TIMEOUT` | `45` | Seconds before an upstream call gives up. |
| `SYSTEM_PROMPT` | see `config.py` | How the agent behaves. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS allowlist. |

## Tools

The agent decides on its own when to call these. All four work without any
extra API key, so a fresh clone has working tool calls immediately.

| Tool | Used for |
| --- | --- |
| `get_current_time` | The date, the day, the time in a given UTC offset. |
| `calculate` | Arithmetic. Parsed as an AST and evaluated node by node — the model cannot smuggle Python through it. |
| `get_weather` | Current conditions and today's range, via Open-Meteo. |
| `search_wikipedia` | A factual summary of a person, place or thing. |

Adding one is a two-step job: describe it in `TOOL_SCHEMAS` and add the
handler to `_REGISTRY`, both in `voice_agent/tools.py`.

## API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | GET | Which stages are configured, and which models. Add `?verify=1` to ask each provider whether it actually accepts its key. |
| `/api/transcribe` | POST | `multipart/form-data` with an `audio` file → `{ text, ms }`. |
| `/api/chat` | POST | `{ messages }` → `{ text, tools, messages, ms }`. |
| `/api/speak` | POST | `{ text, voice_id, speed }` → `audio/mpeg` bytes. |
| `/api/converse` | POST | Think and speak in one round trip; audio comes back base64-encoded. |
| `/api/voices` | GET | Voices available on your ElevenLabs account. |
| `/api/docs` | GET | Interactive OpenAPI documentation. |

The browser app uses the separate `transcribe` → `chat` → `speak` calls so it
can show each stage finishing. `/api/converse` exists for clients that would
rather make one call.

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the weather in Lahore?"}]}'
```

## Deploying

The app is set up for Vercel: `public/` is served as static files and
`api/index.py` runs as a Python serverless function.

```bash
npm i -g vercel
vercel link
vercel deploy --prod
```

Then add `GROQ_API_KEY` (and `ELEVENLABS_API_KEY`, if you want the good voice)
under **Project → Settings → Environment Variables**, and redeploy so the
function picks them up. `/api/health` reports which keys arrived.

## Project structure

```
voice_agent/          the pipeline, independent of any interface
  config.py           environment settings; missing keys fail late, not at import
  asr.py              Groq Whisper transcription
  llm.py              chat completions and the tool-calling loop
  tts.py              ElevenLabs synthesis
  tools.py            tool schemas and their implementations
api/index.py          FastAPI app — uvicorn locally, a function on Vercel
public/               the browser app: index.html, styles.css, app.js
cli.py                terminal agent
tests/                tests for the parts that need no API key
```

## Tests

```bash
pip install pytest
python -m pytest -q
```

They cover the tool layer — including that `calculate` refuses anything that
is not arithmetic — and need no network access or keys.

## Troubleshooting

**The banner says a key is missing.** The server has no `GROQ_API_KEY`. Locally
that means `.env`; on Vercel it means the project's environment variables,
followed by a redeploy.

**The keys are set but every request fails.** Ask the app which one is wrong:

```bash
curl "http://localhost:8000/api/health?verify=1"
```

`keys` reports whether each value has the prefix its provider issues — Groq
keys begin `gsk_`, ElevenLabs keys begin `sk_`, and putting each in the other's
variable is an easy mistake to make. `verified` reports what the providers
themselves say. Neither field exposes any part of a key.

Note that `vercel env add` with no argument prompts for the *name* first, so
pasting a key straight in creates a variable named after your key. Pass the
name as an argument — `vercel env add GROQ_API_KEY production` — and it only
asks for the value.

**It cuts me off mid-sentence.** Raise *Auto-stop after silence* in the
settings panel. 3 s suits a slower speaker.

**It never stops recording.** The room is loud enough that the noise floor
swallowed your voice. Move closer to the microphone, or use the button
instead of waiting for the automatic stop.

**The microphone is blocked.** Allow it for the site and reload; browsers only
ask once. On a deployed copy, check the address really is `https://`.

**Replies are text-only.** Either there is no ElevenLabs key on the server or
*Always use the browser voice* is switched on in settings.

## Licence

MIT — see [LICENSE](LICENSE).
