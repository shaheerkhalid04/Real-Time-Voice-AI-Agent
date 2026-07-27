"""Recovering the tool call Llama sometimes writes as plain text.

Groq rejects the generation instead of parsing it, but the text still names
the tool and carries valid JSON arguments, so the call can be rebuilt.
"""

from __future__ import annotations

import json

from voice_agent.llm import _salvage_tool_call


def test_salvages_the_format_groq_reported():
    generation = '<function=get_current_time({"utc_offset_hours": 5})</function>'
    message = _salvage_tool_call(generation)

    assert message is not None
    call = message["tool_calls"][0]
    assert call["function"]["name"] == "get_current_time"
    assert json.loads(call["function"]["arguments"]) == {"utc_offset_hours": 5}
    assert message["role"] == "assistant"
    assert call["type"] == "function"


def test_salvages_without_the_wrapping_parentheses():
    message = _salvage_tool_call('<function=get_weather{"location": "Lahore"}</function>')
    assert message is not None
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "location": "Lahore"
    }


def test_salvages_when_the_closing_tag_is_missing():
    message = _salvage_tool_call('<function=calculate({"expression": "2 + 2"})')
    assert message is not None
    assert message["tool_calls"][0]["function"]["name"] == "calculate"


def test_refuses_a_tool_that_does_not_exist():
    assert _salvage_tool_call('<function=rm_rf({"path": "/"})</function>') is None


def test_refuses_arguments_that_are_not_json():
    assert _salvage_tool_call("<function=calculate({not json})</function>") is None


def test_returns_none_for_ordinary_text():
    assert _salvage_tool_call("I think the weather is nice today.") is None
    assert _salvage_tool_call("") is None
