# app/agent2_ollama.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "functiongemma:latest"

# Function schema for Agent 2 (Planner)
PLANNER_FUNCTION: Dict[str, Any] = {
    "name": "generate_sop_plan",
    "description": "Generate an SOP-grounded action plan for an industrial anomaly. Severity is fixed by policy.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sop_refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "sections": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "sections"],
                },
            },
            "action_plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer"},
                        "tool": {"type": "string"},
                        "args": {"type": "object"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["step", "tool", "args", "rationale"],
                },
            },
            "required_logging": {
                "type": "object",
                "properties": {
                    "fields": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["fields"],
            },
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "sop_refs", "action_plan", "required_logging", "assumptions"],
    },
}


def _coerce_arguments(args: Any) -> Dict[str, Any]:
    # Ollama may return arguments as dict or JSON string (varies by version/model)
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        return json.loads(args)
    raise TypeError(f"Unexpected tool call arguments type: {type(args)}")


def functiongemma_plan(
    *,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    """
    Calls Ollama functiongemma and returns the structured plan dict (function arguments).
    Raises ValueError if no tool call is returned.
    """
    system_prompt = system_prompt or (
        "You are AnomEdge Planner Agent. "
        "Severity is fixed by policy and must not be changed. "
        "You must follow SOP grounding and respond only via the provided function call."
    )

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "tools": [
            {"type": "function", "function": PLANNER_FUNCTION},
        ],
        "tool_choice": {  # ✅ add this
            "type": "function",
            "function": {"name": "generate_sop_plan"},
        },
        "options": {"temperature": temperature},
    }

    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    msg = data.get("message", {})
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        # Some builds may place info in "content" instead; fail fast so caller can retry/fallback.
        raise ValueError(f"No tool_calls returned. message.content={msg.get('content')!r}")

    call0 = tool_calls[0]
    args = call0.get("function", {}).get("arguments")
    if args is None:
        raise ValueError(f"Tool call missing function.arguments: {call0}")

    return _coerce_arguments(args)