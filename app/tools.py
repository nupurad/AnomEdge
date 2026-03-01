# app/tools.py
from __future__ import annotations

from typing import Any, Dict, Callable, Optional

from app.db import add_audit_event
from app.voice import speak_local  # uses macOS `say` under the hood

def emergency_stop(*, incident_id: str, **kwargs) -> None:
    print("EMERGENCY STOP triggered")

def halt_machine(*, incident_id: str, machine_id: Optional[str] = None, **kwargs) -> None:
    print(f"Halting machine {machine_id or '(unknown)'}")

def pause_conveyor(*, incident_id: str, conveyor_id: Optional[str] = None, **kwargs) -> None:
    print(f"Pausing conveyor {conveyor_id or '(unknown)'}")

def local_alarm(*, incident_id: str, level: str = "P1", **kwargs) -> None:
    print(f"Local alarm activated at level {level}")

def evacuate_radius(*, incident_id: str, meters: int = 10, **kwargs) -> None:
    print(f"Evacuate within {meters} meters")

def deploy_containment(*, incident_id: str, kit: str = "spill_kit", **kwargs) -> None:
    print(f"Deploying containment kit: {kit}")

def clear_obstruction(*, incident_id: str, **kwargs) -> None:
    print("Clearing obstruction (demo action)")

def inspect_belt(*, incident_id: str, **kwargs) -> None:
    print("Inspecting belt (demo action)")

def notify_supervisor(*, incident_id: str, channel: str = "sms", **kwargs) -> None:
    print(f"Notifying supervisor via {channel}")

def notify_safety_officer(*, incident_id: str, channel: str = "sms", **kwargs) -> None:
    print(f"Notifying safety officer via {channel}")

def notify_operations_head(*, incident_id: str, channel: str = "sms", **kwargs) -> None:
    print(f"Notifying operations head via {channel}")

def notify_emergency_response(*, incident_id: str, channel: str = "sms", **kwargs) -> None:
    print(f"Notifying emergency response via {channel}")

def log_checkpoint(*, incident_id: str, note: str = "", **kwargs) -> None:
    print(f"Log checkpoint: {note}")

def voice_announce(
    *,
    incident_id: str,
    message: str,
    severity: str = "P1",
    repeat: int = 1,
    **kwargs
) -> None:
    print(f"[VOICE][{severity}] {message} (x{repeat})")
    speak_local(message, repeat=repeat)


# -----------------------------
# Registry
# -----------------------------

TOOL_REGISTRY: Dict[str, Callable[..., None]] = {
    "emergency_stop": emergency_stop,
    "halt_machine": halt_machine,
    "pause_conveyor": pause_conveyor,
    "local_alarm": local_alarm,
    "voice_announce": voice_announce,
    "evacuate_radius": evacuate_radius,
    "deploy_containment": deploy_containment,
    "clear_obstruction": clear_obstruction,
    "inspect_belt": inspect_belt,
    "notify_supervisor": notify_supervisor,
    "notify_safety_officer": notify_safety_officer,
    "notify_operations_head": notify_operations_head,
    "notify_emergency_response": notify_emergency_response,
    "log_checkpoint": log_checkpoint,
}


# -----------------------------
# Executor
# -----------------------------

def execute_action_step(*, incident_id: str, step: Dict[str, Any]) -> None:
    """
    Executes a single action_plan step and logs an audit event.
    Step schema expected:
      {"step": int, "tool": str, "args": dict, "rationale": str}
    """
    tool_name = step.get("tool")
    args = step.get("args") or {}
    rationale = step.get("rationale", "")

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")

    add_audit_event(
        incident_id=incident_id,
        event_type="tool_executed",
        data={"tool": tool_name, "args": args, "rationale": rationale, "step": step.get("step")},
    )

    # Inject incident_id into all tool calls for consistent logging
    TOOL_REGISTRY[tool_name](incident_id=incident_id, **args)


def execute_action_plan(*, incident_id: str, plan: Dict[str, Any]) -> None:
    """
    Executes plan["action_plan"] in order.
    """
    steps = plan.get("action_plan") or []
    for step in steps:
        execute_action_step(incident_id=incident_id, step=step)