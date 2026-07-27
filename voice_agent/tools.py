"""Tools the agent can call mid-conversation.

Every tool here is keyless — Open-Meteo and Wikipedia both allow anonymous
requests — so a fresh clone gets working tool calls with nothing but a Groq
key. Results are phrased for speech, since whatever comes back is fed to a
model whose reply gets read aloud.
"""

from __future__ import annotations

import ast
import asyncio
import operator
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from urllib.parse import quote

import httpx

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current date and time. Use for any question about "
                "today's date, the day of the week, or what time it is."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "utc_offset_hours": {
                        "type": "number",
                        "description": (
                            "Hours offset from UTC for the place being asked "
                            "about, e.g. 5 for Pakistan, -5 for New York in "
                            "winter. Defaults to UTC."
                        ),
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate an arithmetic expression. Use this for any sum, "
                "percentage, conversion or comparison of numbers instead of "
                "doing the arithmetic yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A maths expression, e.g. '18500 * 0.075'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather and today's forecast for a place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Lahore' or 'Tokyo'.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": (
                "Look up a factual summary of a person, place, organisation, "
                "event or concept. Use it when asked about something you are "
                "unsure of or that may have changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look up, e.g. 'James Webb Space Telescope'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# --------------------------------------------------------------------------
# get_current_time
# --------------------------------------------------------------------------

async def _get_current_time(utc_offset_hours: float = 0.0) -> str:
    try:
        offset = timedelta(hours=float(utc_offset_hours))
    except (TypeError, ValueError):
        offset = timedelta(0)

    now = datetime.now(timezone.utc).astimezone(timezone(offset))
    label = "UTC" if not offset else f"UTC{offset.total_seconds() / 3600:+g}"
    return (
        f"{now.strftime('%A, %d %B %Y')} at "
        f"{now.strftime('%I:%M %p').lstrip('0')} ({label})"
    )


# --------------------------------------------------------------------------
# calculate
# --------------------------------------------------------------------------

# An AST walk rather than eval(): the model controls this string, so only the
# node types listed here are ever executed.
_BINARY_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if type(node.op) is ast.Pow and (abs(right) > 100 or abs(left) > 1e6):
            raise ValueError("that power is too large to compute")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


async def _calculate(expression: str = "") -> str:
    cleaned = (expression or "").strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if not cleaned:
        return "No expression was given."
    if len(cleaned) > 200:
        return "That expression is too long to evaluate."

    try:
        result = _eval_node(ast.parse(cleaned, mode="eval"))
    except ZeroDivisionError:
        return "That divides by zero, so it has no answer."
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        return f"Could not evaluate '{expression}': {exc}."

    if isinstance(result, float):
        result = round(result, 6)
        if result.is_integer():
            result = int(result)
    return f"{expression} = {result:,}"


# --------------------------------------------------------------------------
# get_weather  (Open-Meteo, no API key)
# --------------------------------------------------------------------------

_WEATHER_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers", 95: "a thunderstorm",
    96: "a thunderstorm with hail", 99: "a severe thunderstorm with hail",
}


async def _get_weather(location: str = "") -> str:
    place = (location or "").strip()
    if not place:
        return "No location was given."

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": place, "count": 1, "language": "en", "format": "json"},
            )
            results = geo.json().get("results") if geo.status_code == 200 else None
            if not results:
                return f"Could not find a place called {place}."

            spot = results[0]
            forecast = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": spot["latitude"],
                    "longitude": spot["longitude"],
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
                    "daily": "temperature_2m_max,temperature_2m_min",
                    "timezone": "auto",
                    "forecast_days": 1,
                },
            )
            if forecast.status_code != 200:
                return f"The weather service is unavailable for {place} right now."
            data = forecast.json()
    except Exception:
        return f"Could not reach the weather service for {place}."

    current = data.get("current", {})
    daily = data.get("daily", {})
    name = ", ".join(
        part for part in (spot.get("name"), spot.get("country")) if part
    )
    condition = _WEATHER_CODES.get(current.get("weather_code"), "unclear conditions")

    parts = [
        f"In {name} it is currently {round(current.get('temperature_2m', 0))} degrees "
        f"Celsius with {condition}",
        f"feels like {round(current.get('apparent_temperature', 0))} degrees",
        f"humidity {round(current.get('relative_humidity_2m', 0))} percent",
        f"wind {round(current.get('wind_speed_10m', 0))} kilometres per hour",
    ]
    if daily.get("temperature_2m_max"):
        parts.append(
            f"today's range is {round(daily['temperature_2m_min'][0])} to "
            f"{round(daily['temperature_2m_max'][0])} degrees"
        )
    return ". ".join(parts) + "."


# --------------------------------------------------------------------------
# search_wikipedia
# --------------------------------------------------------------------------

async def _search_wikipedia(query: str = "") -> str:
    term = (query or "").strip()
    if not term:
        return "No search term was given."

    headers = {"User-Agent": "RealTimeVoiceAIAgent/1.0 (educational project)"}
    try:
        async with httpx.AsyncClient(timeout=12.0, headers=headers, follow_redirects=True) as client:
            search = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "list": "search", "srsearch": term,
                    "srlimit": 1, "format": "json",
                },
            )
            hits = search.json().get("query", {}).get("search", []) if search.status_code == 200 else []
            if not hits:
                return f"Wikipedia has no article matching {term}."

            title = hits[0]["title"]
            summary = await client.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + quote(title.replace(" ", "_"), safe="")
            )
            if summary.status_code != 200:
                return f"Could not load the Wikipedia article for {title}."
            extract = (summary.json().get("extract") or "").strip()
    except Exception:
        return f"Could not reach Wikipedia for {term}."

    if not extract:
        return f"The Wikipedia article for {title} has no summary."
    if len(extract) > 900:
        extract = extract[:900].rsplit(".", 1)[0] + "."
    return f"Wikipedia on {title}: {extract}"


_REGISTRY: dict[str, Callable[..., Any]] = {
    "get_current_time": _get_current_time,
    "calculate": _calculate,
    "get_weather": _get_weather,
    "search_wikipedia": _search_wikipedia,
}


async def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name and return a spoken-friendly result string."""
    handler = _REGISTRY.get(name)
    if handler is None:
        return f"There is no tool called {name}."

    try:
        return await asyncio.wait_for(handler(**(arguments or {})), timeout=15.0)
    except asyncio.TimeoutError:
        return f"The {name} tool took too long to respond."
    except TypeError as exc:
        return f"The {name} tool was called with the wrong arguments: {exc}."
    except Exception as exc:  # pragma: no cover - defensive
        return f"The {name} tool failed: {exc}."
