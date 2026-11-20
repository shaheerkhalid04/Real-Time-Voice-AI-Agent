"""The switch to openai/gpt-oss-120b: request shape, reasoning handling, verified health."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from voice_agent import config as config_module
from voice_agent import diagnostics, llm


class FakeResponse:
    def __init__(self, status: int, body: dict[str, Any]):
        self.status_code = status
        self._body = body
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._body


class FakeClient:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.sent: dict[str, Any] | None = None

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> FakeResponse:
        self.sent = json
        return self.response


def settings(model: str = "openai/gpt-oss-120b"):
    return dataclasses.replace(config_module.get_settings(), llm_model=model, groq_api_key="test")


def test_default_model_is_not_retired():
    assert config_module.Settings().llm_model == "openai/gpt-oss-120b"


def test_gpt_oss_requests_ask_for_low_reasoning_and_strip_it_from_replays():
    reply = {"choices": [{"message": {"role": "assistant", "content": "Hi", "reasoning": "secret", "tool_calls": None}}]}
    client = FakeClient(FakeResponse(200, reply))
    message = asyncio.run(llm._complete(client, [{"role": "user", "content": "hi"}], "key", settings(), True))
    assert client.sent is not None
    assert client.sent["reasoning_effort"] == "low"
    assert client.sent["tool_choice"] == "auto" and client.sent["tools"]
    assert "reasoning" not in message and "tool_calls" not in message


def test_other_models_get_no_reasoning_parameter():
    client = FakeClient(FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": "Hi"}}]}))
    asyncio.run(llm._complete(client, [{"role": "user", "content": "hi"}], "key", settings("some-other-model"), False))
    assert client.sent is not None and "reasoning_effort" not in client.sent


def test_verify_model_reports_a_retired_model(monkeypatch):
    class Gone:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return FakeClient(FakeResponse(404, {"error": {"code": "model_not_found"}}))

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(diagnostics.httpx, "AsyncClient", Gone)
    result = asyncio.run(diagnostics.verify_model(settings()))
    assert result["ok"] is False and "latency_ms" not in result


def test_earlier_turns_drop_their_tool_traffic():
    tool_turn = [
        {"role": "user", "content": "Weather in Lahore?"},
        {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "name": "get_weather", "content": "24 C, clear"},
        {"role": "assistant", "content": "About 24 degrees and clear."},
    ]
    current = [
        {"role": "user", "content": "What is 18500 times 0.075?"},
        {"role": "assistant", "tool_calls": [{"id": "b", "type": "function", "function": {"name": "calculate", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "name": "calculate", "content": "1,387.5"},
    ]
    history = llm._with_system_prompt([*tool_turn, *current], "be brief")

    assert [m["role"] for m in history] == ["system", "user", "assistant", "user", "assistant", "tool"]
    assert history[2]["content"] == "About 24 degrees and clear."
    assert history[3:] == current


def test_history_without_a_user_turn_is_left_alone():
    assert llm._earlier_turns_as_text([{"role": "assistant", "content": "hi"}]) == [{"role": "assistant", "content": "hi"}]
