from app.classify_severity import Signals, classify_severity
from app.db import count_p1_last_30_min, has_recurring_within_24h

def apply_policy(model_output: dict, *, zone: str | None, machine_id: str | None):
    flags = model_output.get("flags", {})

    signals = Signals(
        injury_risk=flags.get("injury_risk", False),
        is_spreading=flags.get("is_spreading", False),
        hazard_suspected=flags.get("hazard_suspected", False),
        conveyor_halted=flags.get("conveyor_halted", False),
        motor_overheating=flags.get("motor_overheating", False),
        belt_damage_visible=flags.get("belt_damage_visible", False),
    )

    anomaly = model_output.get("anomaly_type", "normal")

    p1_count = count_p1_last_30_min(zone=zone, machine_id=machine_id)
    recurring = has_recurring_within_24h(
        anomaly_type=anomaly,
        zone=zone,
        machine_id=machine_id,
    )

    severity, tags = classify_severity(
        anomaly_raw=anomaly,
        signals=signals,
        p1_events_last_30_min=p1_count,
        recurring_within_24h=recurring,
    )

    return severity, tags