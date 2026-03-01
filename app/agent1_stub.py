from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List


def agent1_stub_from_scenario(
    *,
    scenario: str,
    observations: List[str] | None = None,
) -> Dict[str, Any]:
    base = {
        "frame_id": uuid.uuid4().hex[:8],
        "timestamp": int(time.time()),
        "anomaly_type": "normal",
        "confidence": 0.95,
        "flags": {
            "injury_risk": False,
            "is_spreading": False,
            "hazard_suspected": False,
            "conveyor_halted": False,
            "motor_overheating": False,
            "belt_damage_visible": False,
        },
        "evidence": {"observations": observations or [], "bbox": []},
    }

    s = (scenario or "").lower().strip()
    if s == "smoke":
        base["anomaly_type"] = "smoke_fire"
        base["confidence"] = 0.93
        base["flags"]["injury_risk"] = True
        base["evidence"]["observations"] = base["evidence"]["observations"] or ["visible smoke near motor"]
    elif s == "leak":
        base["anomaly_type"] = "oil_leak"
        base["confidence"] = 0.86
        base["flags"]["is_spreading"] = True
        base["evidence"]["observations"] = base["evidence"]["observations"] or ["dark reflective fluid spreading"]
    elif s == "jam":
        base["anomaly_type"] = "conveyor_jam"
        base["confidence"] = 0.88
        base["flags"]["conveyor_halted"] = True
        base["evidence"]["observations"] = base["evidence"]["observations"] or ["belt stopped; obstruction visible"]

    return base
