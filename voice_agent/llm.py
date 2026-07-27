"""The agent brain: Groq chat completions plus a tool-calling loop.

`respond()` takes the conversation so far and returns the reply text along
with a record of every tool the model decided to call, so the UI can show
what happened rather than just the final sentence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import GROQ_BASE_URL, Settings, get_settings
from .tools import TOOL_SCHEMAS, run_tool

# Llama intermittently emits a tool call in its old pseudo-XML text format
# instead of structured JSON, and Groq rejects the generation rather than
# parsing it. The arguments are usually correct, so the call is recoverable:
#     <function=get_current_time({"utc_offset_hours": 5})</function>
_TEXT_TOOL_CALL = re.compile(
    r"<function=([A-Za-z_][A-Za-z0-9_]*)\s*\(?\s*(\{.*?\})\s*\)?\s*(?:</function>|$)",
    re.DOTALL,
)

_TOOL_NAMES = {schema["function"]["name"] for schema in TOOL_SCHEMAS}

# How many times to resample when the model produces an unparseable call.
_FORMAT_RETRIES = 2


class LLMError(RuntimeError):
    """The language model call failed."""


class ToolCallFormatError(LLMError):
    """Groq refused the model's tool call because it was malformed."""

    def __init__(self, message: str, failed_generation: str = "") -> None:
        super().__init__(message)
        self.failed_generation = failed_generation


@dataclass
class ToolInvocation:
    name: str
    arguments: dict[str, Any]
    result: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "result": self.result}


@dataclass
class AgentReply:
    text: str
    tools: list[ToolInvocation] = field(default_factory=list)
    # The full message list including tool traffic, so the caller can persist
    # an accurate history and the model keeps its own context next turn.
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tools": [tool.to_dict() for tool in self.tools],
            "messages": self.messages,
        }


async def respond(
    messages: list[dict[str, Any]],
    *,
    use_tools: bool = True,
    system_prompt: str | None = None,
    settings: Settings | None = None,
) -> AgentReply:
    """Run one conversational turn, resolving any tool calls along the way."""
    settings = settings or get_settings()
    api_key = settings.require_groq()

    history = _with_system_prompt(messages, system_prompt or settings.system_prompt)
    invocations: list[ToolInvocation] = []

    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        for round_index in range(settings.max_tool_rounds + 1):
            # On the final round the tools are withheld, which forces the model
            # to answer with words instead of asking for yet another call.
            offer_tools = use_tools and round_index < settings.max_tool_rounds
            message = await _complete_resiliently(
                client, history, api_key, settings, offer_tools
            )
            history.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                text = (message.get("content") or "").strip()
                if not text:
                    text = "Sorry, I did not catch that. Could you say it again?"
                return AgentReply(text=text, tools=invocations, messages=history)

            for call in tool_calls:
                invocation = await _dispatch(call)
                invocations.append(invocation)
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "name": invocation.name,
                        "content": invocation.result,
                    }
                )

    # Unreachable in practice: the loop above always returns on the last round.
    return AgentReply(
        text="I ran into trouble working that out. Could you ask again?",
        tools=invocations,
        messages=history,
    )


async def _complete_resiliently(
    client: httpx.AsyncClient,
    history: list[dict[str, Any]],
    api_key: str,
    settings: Settings,
    offer_tools: bool,
) -> dict[str, Any]:
    """Complete a turn, surviving the model's malformed tool calls.

    Resampling usually fixes it. If it does not, the botched text still names
    the tool and its arguments, so the call can be rebuilt rather than lost.
    As a last resort the turn is answered without tools, because a plain
    answer beats an error message.
    """
    last_error: ToolCallFormatError | None = None

    for _ in range(_FORMAT_RETRIES + 1):
        try:
            return await _complete(client, history, api_key, settings, offer_tools)
        except ToolCallFormatError as exc:
            last_error = exc

    if last_error is not None:
        salvaged = _salvage_tool_call(last_error.failed_generation)
        if salvaged is not None:
            return salvaged

    return await _complete(client, history, api_key, settings, offer_tools=False)


def _salvage_tool_call(failed_generation: str) -> dict[str, Any] | None:
    """Rebuild a proper tool-call message from the model's text-format call."""
    match = _TEXT_TOOL_CALL.search(failed_generation or "")
    if match is None:
        return None

    name, raw_arguments = match.group(1), match.group(2)
    if name not in _TOOL_NAMES:
        return None

    try:
        json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None

    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": f"salvaged_{name}",
                "type": "function",
                "function": {"name": name, "arguments": raw_arguments},
            }
        ],
    }


async def _complete(
    client: httpx.AsyncClient,
    history: list[dict[str, Any]],
    api_key: str,
    settings: Settings,
    offer_tools: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": history,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    if offer_tools:
        payload["tools"] = TOOL_SCHEMAS
        payload["tool_choice"] = "auto"

    try:
        response = await client.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    except httpx.RequestError as exc:
        raise LLMError(f"Could not reach the language model: {exc}") from exc

    if response.status_code != 200:
        botched = _failed_generation(response)
        if botched is not None:
            raise ToolCallFormatError(_describe_error(response), botched)
        raise LLMError(_describe_error(response))

    try:
        message = response.json()["choices"][0]["message"]
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError("The language model returned an unexpected response.") from exc

    # Strip nulls so the message can be replayed back to the API verbatim.
    return {key: value for key, value in message.items() if value is not None}


async def _dispatch(call: dict[str, Any]) -> ToolInvocation:
    function = call.get("function", {})
    name = function.get("name", "unknown")

    raw_args = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        if not isinstance(arguments, dict):
            arguments = {}
    except json.JSONDecodeError:
        return ToolInvocation(
            name=name,
            arguments={},
            result=f"The arguments for {name} were not valid JSON.",
        )

    result = await run_tool(name, arguments)
    return ToolInvocation(name=name, arguments=arguments, result=result)


def _with_system_prompt(
    messages: list[dict[str, Any]], system_prompt: str
) -> list[dict[str, Any]]:
    history = [dict(message) for message in messages if message.get("role") != "system"]
    return [{"role": "system", "content": system_prompt}, *history]


def _failed_generation(response: httpx.Response) -> str | None:
    """Return the model's rejected output, if this was a tool-format failure."""
    try:
        error = response.json().get("error", {})
    except Exception:
        return None
    botched = error.get("failed_generation")
    return str(botched) if botched else None


def _describe_error(response: httpx.Response) -> str:
    if response.status_code == 401:
        return "Groq rejected the API key. Check GROQ_API_KEY."
    if response.status_code == 429:
        return "Groq rate limit reached. Wait a moment and try again."
    try:
        error = response.json().get("error", {})
        message = error.get("message")
        # Groq puts the model's malformed output here when it rejects a tool
        # call. Without it there is no way to tell which tool went wrong.
        botched = error.get("failed_generation")
        if message and botched:
            return f"The language model failed: {message} | generated: {str(botched)[:400]}"
        if message:
            return f"The language model failed: {message}"
    except Exception:
        pass
    return f"The language model failed with HTTP {response.status_code}."
