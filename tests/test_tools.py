"""Tests for the parts that run without any API key.

    pip install pytest
    python -m pytest -q
"""

from __future__ import annotations

import asyncio

import pytest

from voice_agent.tools import TOOL_SCHEMAS, run_tool


def call(name: str, **arguments) -> str:
    return asyncio.run(run_tool(name, arguments))


# --------------------------------------------------------------- calculate

@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", "4"),
        ("18500 * 0.075", "1,387.5"),
        ("2 ** 10", "1,024"),
        ("(9 + 3) / 4", "3"),
        ("10 ^ 2", "100"),
    ],
)
def test_calculate_returns_the_answer(expression, expected):
    assert expected in call("calculate", expression=expression)


def test_calculate_reports_division_by_zero():
    assert "zero" in call("calculate", expression="5 / 0").lower()


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "[x for x in range(10)]",
        "9 ** 9 ** 9",
    ],
)
def test_calculate_refuses_anything_that_is_not_arithmetic(expression):
    assert "could not evaluate" in call("calculate", expression=expression).lower()


# ---------------------------------------------------------------- the rest

def test_current_time_honours_the_offset():
    utc = call("get_current_time")
    pakistan = call("get_current_time", utc_offset_hours=5)
    assert "UTC" in utc and "UTC+5" in pakistan


def test_unknown_tool_is_reported_not_raised():
    assert "no tool called" in call("wipe_disk").lower()


def test_tools_are_called_with_the_wrong_arguments_safely():
    assert "wrong arguments" in call("calculate", nonsense=1).lower()


def test_every_advertised_tool_is_implemented():
    for schema in TOOL_SCHEMAS:
        name = schema["function"]["name"]
        assert "no tool called" not in call(name).lower(), name
