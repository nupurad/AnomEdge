from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, List

Severity = Literal["P0", "P1", "P2"]
Anomaly = Literal["smoke_fire", "oil_leak", "belt_damage", "normal"]

@dataclass
class Signals:
    injury_risk: bool = False
    # leak
    is_spreading: bool = False
    hazard_suspected: bool = False
    # jam
    conveyor_halted: bool = False
    motor_overheating: bool = False
    belt_damage_visible: bool = False

def normalize_anomaly(a: str) -> Anomaly:
    a = (a or "").strip().lower()
    if a in ("smoke", "fire", "smoke_fire"):
        return "smoke_fire"
    if a in ("oil_leak", "leak", "fluid_leak"):
        return "oil_leak"
    if a in ("belt_damage", "conveyor_jam", "jam", "tear", "wear"):
        return "belt_damage"
    return "normal"

def base_severity(anomaly: Anomaly, s: Signals) -> Severity:
    # SOP-FIRE-002: any visible smoke or fire -> P0 :contentReference[oaicite:1]{index=1}
    if anomaly == "smoke_fire":
        return "P0"

    # SOP-LEAK-001 ladder :contentReference[oaicite:2]{index=2}
    if anomaly == "oil_leak":
        if s.hazard_suspected:
            return "P0"
        if s.is_spreading:
            return "P1"
        return "P2"

    # SOP-JAM-003 ladder :contentReference[oaicite:3]{index=3}
    if anomaly == "belt_damage":
        if s.motor_overheating or s.belt_damage_visible:
            return "P0"
        if s.conveyor_halted:
            return "P1"
        return "P2"

    return "P2"

def classify_severity(
    anomaly_raw: str,
    signals: Signals,
    *,
    p1_events_last_30_min: int = 0,
    recurring_within_24h: bool = False,
) -> Tuple[Severity, List[str]]:
    """
    Returns (severity, policy_tags) where policy_tags can be passed to Agent 2
    and stored in plan_json/audit_events.
    """
    tags: List[str] = []

    anomaly = normalize_anomaly(anomaly_raw)
    sev = base_severity(anomaly, signals)

    # SOP-SAF-004 escalation rules :contentReference[oaicite:4]{index=4}
    if signals.injury_risk:
        sev = "P0"
        tags.append("injury_risk->P0")

    if sev == "P1" and p1_events_last_30_min >= 2:
        sev = "P0"
        tags.append("two_P1_30min->P0")

    # SOP-SAF-004 recurring issue rule (not necessarily severity) :contentReference[oaicite:5]{index=5}
    if recurring_within_24h:
        tags.append("maintenance_required_24h_recurrence")

    return sev, tags
