"""The agent brain: Groq chat completions plus a tool-calling loop.

`respond()` takes the conversation so far and returns the reply text along
with a record of every tool the model decided to call, so the UI can show
what happened rather than just the final sentence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import GROQ_BASE_URL, Settings, get_settings
from .tools import TOOL_SCHEMAS, run_tool


class LLMError(RuntimeError):
    """The language model call failed."""


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
            message = await _complete(client, history, api_key, settings, offer_tools)
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


def _describe_error(response: httpx.Response) -> str:
    if response.status_code == 401:
        return "Groq rejected the API key. Check GROQ_API_KEY."
    if response.status_code == 429:
        return "Groq rate limit reached. Wait a moment and try again."
    try:
        message = response.json().get("error", {}).get("message")
        if message:
            return f"The language model failed: {message}"
    except Exception:
        pass
    return f"The language model failed with HTTP {response.status_code}."
