# app/planner.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.sop_index import SOP, load_sops, retrieve_sops_with_framework
from app.agent2_ollama import functiongemma_plan
from app.db import add_audit_event, update_incident_plan


# Load SOPs once (edge-friendly)
_SOPS: List[SOP] = load_sops("data/sop")


# -----------------------------
# Grounding builder
# -----------------------------

def build_grounding_payload(sops: Sequence[SOP]) -> Dict[str, Any]:
    """
    Keep this compact. Agent 2 should see the actionable sections + severity guidance.
    """
    return {
        "sops": [
            {
                "id": s.sop_id,
                "title": s.title,
                "trigger_conditions": s.triggers,
                "severity_guidance": s.severity_guidance,
                "immediate_actions": s.immediate_actions,
                "escalation_criteria": s.escalation_criteria,
                "required_logging": s.required_logging,
            }
            for s in sops
        ]
    }


# -----------------------------
# Prompting
# -----------------------------

DEFAULT_ALLOWED_TOOLS = [
    # halt / control
    "halt_machine",
    "pause_conveyor",
    "emergency_stop",
    # alerting
    "local_alarm",
    "voice_announce",
    # safety actions
    "evacuate_radius",
    "deploy_containment",
    "clear_obstruction",
    "inspect_belt",
    "resume_test_cycle",
    # notifications
    "notify_supervisor",
    "notify_safety_officer",
    "notify_operations_head",
    "notify_emergency_response",
    # logging
    "log_checkpoint",
]

def build_agent2_user_prompt(
    *,
    anomaly_type: str,
    severity: str,
    confidence: float,
    observations: List[str],
    grounding_payload: Dict[str, Any],
    policy_tags: Optional[List[str]] = None,
    allowed_tools: Optional[List[str]] = None,
) -> str:
    allowed_tools = allowed_tools or DEFAULT_ALLOWED_TOOLS
    policy_tags = policy_tags or []

    return f"""
You are EdgeSentinel Planner Agent.
Severity is fixed by policy: {severity}. Do NOT change severity.
You MUST ground every action in SOP_GROUNDING. Do not invent steps.
Use ONLY these tools: {json.dumps(allowed_tools)}

INPUT:
anomaly_type: {anomaly_type}
severity: {severity}
confidence: {confidence}
policy_tags: {json.dumps(policy_tags)}
observations: {json.dumps(observations, ensure_ascii=False)}

SOP_GROUNDING (authoritative):
{json.dumps(grounding_payload, ensure_ascii=False)}

Return the plan by calling the function generate_sop_plan with:
- summary
- sop_refs (include SOP-SAF-004 if present in grounding)
- action_plan (3–8 steps, each with rationale citing SOP id)
- required_logging.fields
- assumptions
""".strip()


# -----------------------------
# Validation / guardrails
# -----------------------------

def validate_plan(
    plan: Dict[str, Any],
    *,
    severity: str,
    anomaly_type: str,
    allowed_tools: Sequence[str],
) -> None:
    # Required top-level keys
    for k in ("summary", "sop_refs", "action_plan", "required_logging", "assumptions"):
        if k not in plan:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(plan["summary"], str) or not plan["summary"].strip():
        raise ValueError("summary must be a non-empty string")

    if not isinstance(plan["sop_refs"], list) or len(plan["sop_refs"]) == 0:
        raise ValueError("sop_refs must be a non-empty list")

    if not isinstance(plan["action_plan"], list) or len(plan["action_plan"]) == 0:
        raise ValueError("action_plan must be a non-empty list")

    rl = plan["required_logging"]
    if not isinstance(rl, dict) or "fields" not in rl or not isinstance(rl["fields"], list):
        raise ValueError("required_logging.fields must be a list")

    allowed = set(allowed_tools)
    tools_used = set()

    for step in plan["action_plan"]:
        if not isinstance(step, dict):
            raise ValueError("action_plan steps must be objects")
        for k in ("step", "tool", "args", "rationale"):
            if k not in step:
                raise ValueError(f"action_plan step missing {k}")
        if not isinstance(step["args"], dict):
            raise ValueError("action_plan[].args must be an object")
        if not isinstance(step["rationale"], str) or not step["rationale"].strip():
            raise ValueError("action_plan[].rationale must be a non-empty string")
        tool = step["tool"]
        if tool not in allowed:
            raise ValueError(f"Unknown/unallowed tool: {tool}")
        tools_used.add(tool)

    # Severity/anomaly hard requirements (safety-critical)
    if severity == "P0":
        if not ({"halt_machine", "pause_conveyor", "emergency_stop"} & tools_used):
            raise ValueError("P0 plan must include a halt (halt_machine/pause_conveyor/emergency_stop)")

    if anomaly_type == "smoke_fire":
        if "evacuate_radius" not in tools_used:
            raise ValueError("smoke_fire plan must include evacuate_radius")
        if not ({"local_alarm", "voice_announce"} & tools_used):
            raise ValueError("smoke_fire plan must include local_alarm or voice_announce")


def fallback_plan(*, severity: str, anomaly_type: str) -> Dict[str, Any]:
    actions: List[Dict[str, Any]] = []
    step = 1

    if severity == "P0":
        actions.append({"step": step, "tool": "emergency_stop", "args": {}, "rationale": "Fallback safety halt"}); step += 1
        actions.append({"step": step, "tool": "local_alarm", "args": {"level": "P0"}, "rationale": "Fallback alarm"}); step += 1
        if anomaly_type == "smoke_fire":
            actions.append({"step": step, "tool": "evacuate_radius", "args": {"meters": 10}, "rationale": "Fallback evacuation"}); step += 1
            actions.append({"step": step, "tool": "notify_safety_officer", "args": {}, "rationale": "Fallback notify safety"}); step += 1
    elif severity == "P1":
        actions.append({"step": step, "tool": "notify_supervisor", "args": {}, "rationale": "Fallback notify supervisor"}); step += 1
        actions.append({"step": step, "tool": "log_checkpoint", "args": {}, "rationale": "Fallback log"}); step += 1
    else:
        actions.append({"step": step, "tool": "log_checkpoint", "args": {}, "rationale": "Fallback log"}); step += 1

    return {
        "summary": f"Fallback plan for {anomaly_type} at severity {severity}.",
        "sop_refs": [{"id": "SOP-SAF-004", "sections": ["Logging Requirements"]}],
        "action_plan": actions,
        "required_logging": {"fields": ["timestamp", "location_or_machine_id", "assigned_severity"]},
        "assumptions": ["Planner output invalid or unavailable; using deterministic fallback plan."],
    }

def normalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure required top-level keys exist
    plan.setdefault("summary", "")
    plan.setdefault("sop_refs", [])
    plan.setdefault("action_plan", [])
    plan.setdefault("required_logging", {"fields": []})
    plan.setdefault("assumptions", [])

    # Normalize tool names like "notify safety officer" -> "notify_safety_officer"
    for step in plan.get("action_plan", []) or []:
        if isinstance(step, dict):
            if isinstance(step.get("tool"), str):
                step["tool"] = step["tool"].strip().lower().replace(" ", "_")
            step.setdefault("args", {})
            step.setdefault("rationale", "")

    return plan

# -----------------------------
# Main entrypoint: plan an incident
# -----------------------------

def plan_incident(
    *,
    incident_id: str,
    anomaly_type: str,
    severity: str,
    confidence: float,
    observations: List[str],
    policy_tags: Optional[List[str]] = None,
    allowed_tools: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (plan_dict, notes).
    - Retrieves SOPs
    - Calls FunctionGemma planner
    - Validates
    - Persists to DB (summary, sop_refs_json, plan_json) + audit event
    """
    allowed_tools = allowed_tools or DEFAULT_ALLOWED_TOOLS
    notes: List[str] = []

    # 1) Retrieve SOPs (+ SAF framework)
    matched = retrieve_sops_with_framework(
        _SOPS,
        anomaly_type=anomaly_type,
        severity=severity,
        observations=observations,
        top_k=2,
    )
    grounding = build_grounding_payload(matched)

    add_audit_event(
        incident_id=incident_id,
        event_type="sop_retrieved",
        data={"sop_ids": [s.sop_id for s in matched], "anomaly_type": anomaly_type, "severity": severity},
    )

    # 2) Prompt + function call
    user_prompt = build_agent2_user_prompt(
        anomaly_type=anomaly_type,
        severity=severity,
        confidence=confidence,
        observations=observations,
        grounding_payload=grounding,
        policy_tags=policy_tags,
        allowed_tools=allowed_tools,
    )

    try:
        plan = functiongemma_plan(user_prompt=user_prompt)
        plan = normalize_plan(plan)  # ✅ add this line
        validate_plan(plan, severity=severity, anomaly_type=anomaly_type, allowed_tools=allowed_tools)
    except Exception as e:
        notes.append(f"planner_failed:{type(e).__name__}:{e}")
        plan = fallback_plan(severity=severity, anomaly_type=anomaly_type)

    # 3) Persist
    update_incident_plan(
        incident_id,
        summary=plan.get("summary"),
        sop_refs={"sop_refs": plan.get("sop_refs", [])},
        plan=plan,
    )

    add_audit_event(
        incident_id=incident_id,
        event_type="plan_generated",
        data={
            "notes": notes,
            "sop_ids": [r.get("id") for r in plan.get("sop_refs", []) if isinstance(r, dict)],
            "tools": [s.get("tool") for s in plan.get("action_plan", []) if isinstance(s, dict)],
        },
    )

    return plan, notes