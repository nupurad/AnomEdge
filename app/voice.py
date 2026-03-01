import subprocess
from typing import Optional


TEMPLATES = {
    ("P0", "smoke_fire"): "P0 ALERT. Smoke or fire detected in {zone} near {machine}. Evacuate within 10 meters. Do not resume until cleared.",
    ("P0", "oil_leak"): "P0 ALERT. Hazardous fluid leak suspected in {zone} near {machine}. Keep clear. Operations halted.",
    ("P0", "conveyor_jam"): "P0 ALERT. Conveyor hazard detected in {zone} near {machine}. Keep clear.",
    ("P1", "oil_leak"): "P1 alert. Fluid leak detected in {zone}. Supervisor notified.",
    ("P1", "conveyor_jam"): "P1 alert. Conveyor halted in {zone}. Maintenance required.",
}


def build_announcement(severity: str, anomaly_type: str, zone: Optional[str], machine_id: Optional[str]):
    if severity not in ("P0", "P1"):
        return None

    template = TEMPLATES.get((severity, anomaly_type))
    if not template:
        template = f"{severity} alert in {{zone}} near {{machine}}."

    message = template.format(
        zone=zone or "the affected area",
        machine=machine_id or "equipment",
    )

    repeat = 2 if severity == "P0" else 1

    return {"message": message, "repeat": repeat}


def speak_local(message: str, repeat: int = 1):
    for _ in range(repeat):
        subprocess.run(["say", message])