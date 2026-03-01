# ui/app.py
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from typing import Any, Dict, List, Optional

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from PIL import Image
from app.voice import build_announcement, speak_local
from app.db import (
    init_db,
    insert_incident,
    add_audit_event,
    count_p1_last_30_min,
    has_recurring_within_24h,
)
from app.classify_severity import Signals, classify_severity
from app.planner import plan_incident
from app.tools import execute_action_plan
from app.agent1_stub import agent1_stub_from_scenario

DB_PATH = "data/edge_sentinel.db"


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_incidents(limit: int = 50):
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, created_at, zone, machine_id, anomaly_type, severity, status, confidence
        FROM incidents
        ORDER BY created_at DESC
        LIMIT ?;
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def fetch_incident_detail(incident_id: str):
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM incidents WHERE id = ?;", (incident_id,))
    row = cur.fetchone()
    conn.close()
    return row


def fetch_audit(incident_id: str):
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM audit_events WHERE incident_id = ? ORDER BY timestamp ASC;",
        (incident_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def safe_json_loads(s: str) -> Dict[str, Any]:
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("JSON must be an object at the top level")
    return obj


def compute_policy(
    agent1: Dict[str, Any],
    *,
    zone: Optional[str],
    machine_id: Optional[str],
) -> tuple[str, List[str], Signals]:
    flags = agent1.get("flags") or {}

    signals = Signals(
        injury_risk=bool(flags.get("injury_risk", False)),
        is_spreading=bool(flags.get("is_spreading", False)),
        hazard_suspected=bool(flags.get("hazard_suspected", False)),
        conveyor_halted=bool(flags.get("conveyor_halted", False)),
        motor_overheating=bool(flags.get("motor_overheating", False)),
        belt_damage_visible=bool(flags.get("belt_damage_visible", False)),
    )

    anomaly_type = agent1.get("anomaly_type", "normal")

    p1_count = count_p1_last_30_min(zone=zone, machine_id=machine_id)
    recurring = has_recurring_within_24h(anomaly_type=anomaly_type, zone=zone, machine_id=machine_id)

    sev, tags = classify_severity(
        anomaly_raw=anomaly_type,
        signals=signals,
        p1_events_last_30_min=p1_count,
        recurring_within_24h=recurring,
    )
    return sev, tags, signals


def main():
    st.set_page_config(page_title="EdgeSentinel UI", layout="wide")
    st.title("EdgeSentinel – On-device Anomaly Detection + SOP Planner")

    init_db(reset=False)

    # ----------------------------
    # Sidebar: Input + Controls
    # ----------------------------
    with st.sidebar:
        st.header("Input")

        uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])
        scenario = st.selectbox("Temporary scenario (Agent1 not ready)", ["smoke", "leak", "jam"])
        obs_text = st.text_area("Optional observations (one per line)", value="")
        observations = [ln.strip() for ln in obs_text.splitlines() if ln.strip()]

        st.divider()
        st.header("Context")
        camera_id = st.text_input("camera_id", value="cam-01")
        zone = st.text_input("zone", value="Zone-3")
        machine_id = st.text_input("machine_id", value="Machine-A")
        connectivity = st.selectbox("connectivity", ["offline", "online"])
        model_name = st.text_input("model_name", value="gemma3n-lora-agent1 (stub)")

        st.divider()
        st.header("Advanced")
        use_manual_agent1 = st.checkbox("Paste Agent1 JSON (override stub)", value=False)
        manual_agent1_json = st.text_area(
            "Agent1 JSON",
            value="",
            height=160,
            disabled=not use_manual_agent1,
            placeholder='{"anomaly_type":"smoke_fire","confidence":0.9,"flags":{...},"evidence":{"observations":[],"bbox":[]}}',
        )
        run_tools = st.checkbox("Execute tools (prints + macOS TTS)", value=True)

        run_clicked = st.button("Run full pipeline", type="primary")

    # ----------------------------
    # Main area: Preview image + live outputs
    # ----------------------------
    left, right = st.columns([1, 1])

    image_path = None
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        with left:
            st.subheader("Input image")
            st.image(img, use_container_width=True)
        # Save to a temp file so you can store path in DB
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            image_path = tmp.name

    # ----------------------------
    # Run pipeline
    # ----------------------------
    if run_clicked:
        try:
            # 1) Agent 1 output (stub or manual)
            if use_manual_agent1 and manual_agent1_json.strip():
                agent1 = safe_json_loads(manual_agent1_json.strip())
                # Ensure evidence structure exists
                agent1.setdefault("evidence", {"observations": [], "bbox": []})
                agent1["evidence"].setdefault("observations", [])
                agent1["evidence"].setdefault("bbox", [])
            else:
                agent1 = agent1_stub_from_scenario(scenario=scenario, observations=observations)

            st.session_state["last_agent1"] = agent1

            # 2) Policy (severity + tags)
            severity, policy_tags, signals = compute_policy(agent1, zone=zone, machine_id=machine_id)
            st.session_state["last_policy"] = {"severity": severity, "policy_tags": policy_tags}

            # 2b) Deterministic voice announcement (demo-safe)
            ann = build_announcement(
                severity=severity,
                anomaly_type=agent1.get("anomaly_type", "normal"),
                zone=zone,
                machine_id=machine_id,
            )
            if ann:
                speak_local(ann["message"], repeat=ann["repeat"])

            # 3) Create incident
            incident_id = insert_incident(
                camera_id=camera_id,
                zone=zone,
                machine_id=machine_id,
                anomaly_type=agent1.get("anomaly_type", "normal"),
                severity=severity,
                confidence=float(agent1.get("confidence") or 0.0),
                summary="Initial detection",
                sop_refs=None,
                plan={"agent1": agent1, "policy_tags": policy_tags},
                image_path=image_path,
                model_name=model_name,
                connectivity=connectivity,
            )
            st.session_state["last_incident_id"] = incident_id

            # 4) Audit events
            add_audit_event(incident_id=incident_id, event_type="perception_classified", data=agent1)
            add_audit_event(
                incident_id=incident_id,
                event_type="severity_assigned",
                data={
                    "severity": severity,
                    "policy_tags": policy_tags,
                    "signals": {
                        "injury_risk": signals.injury_risk,
                        "is_spreading": signals.is_spreading,
                        "hazard_suspected": signals.hazard_suspected,
                        "conveyor_halted": signals.conveyor_halted,
                        "motor_overheating": signals.motor_overheating,
                        "belt_damage_visible": signals.belt_damage_visible,
                    },
                },
            )

            # 5) Agent 2 planner (FunctionGemma via Ollama)
            plan, notes = plan_incident(
                incident_id=incident_id,
                anomaly_type=agent1.get("anomaly_type", "normal"),
                severity=severity,
                confidence=float(agent1.get("confidence") or 0.0),
                observations=agent1.get("evidence", {}).get("observations", []) or observations,
                policy_tags=policy_tags,
            )
            st.session_state["last_plan"] = plan
            st.session_state["last_notes"] = notes

            # 6) Execute tools
            if run_tools:
                execute_action_plan(incident_id=incident_id, plan=plan)

            st.success(f"Pipeline complete. Incident: {incident_id}")

        except Exception as e:
            st.error(f"Pipeline failed: {type(e).__name__}: {e}")

    # ----------------------------
    # Left pane: Latest outputs
    # ----------------------------
    with left:
        st.subheader("Agent 1 Output (Perception)")
        st.json(st.session_state.get("last_agent1", {}))

        st.subheader("Policy Output")
        st.json(st.session_state.get("last_policy", {}))

        st.subheader("Agent 2 Plan (SOP-grounded)")
        st.json(st.session_state.get("last_plan", {}))
        if "last_notes" in st.session_state:
            st.caption(f"Planner notes: {st.session_state['last_notes']}")

    # ----------------------------
    # Right pane: DB viewers
    # ----------------------------
    with right:
        st.subheader("Incidents (SQLite)")
        incidents = fetch_incidents(limit=50)
        if incidents:
            st.dataframe(
                [
                    {
                        "id": r["id"],
                        "created_at": r["created_at"],
                        "zone": r["zone"],
                        "machine": r["machine_id"],
                        "anomaly": r["anomaly_type"],
                        "sev": r["severity"],
                        "conf": r["confidence"],
                        "status": r["status"],
                    }
                    for r in incidents
                ],
                use_container_width=True,
            )

            default_id = st.session_state.get("last_incident_id") or incidents[0]["id"]
            selected = st.selectbox(
                "Select incident for details",
                options=[r["id"] for r in incidents],
                index=[r["id"] for r in incidents].index(default_id) if default_id in [r["id"] for r in incidents] else 0,
            )

            st.subheader("Incident detail")
            detail = fetch_incident_detail(selected)
            if detail:
                st.json({k: detail[k] for k in detail.keys()})

            st.subheader("Audit trail")
            audit = fetch_audit(selected)
            st.dataframe(
                [
                    {
                        "timestamp": a["timestamp"],
                        "event_type": a["event_type"],
                        "data_json": a["data_json"],
                    }
                    for a in audit
                ],
                use_container_width=True,
            )
        else:
            st.info("No incidents yet. Use the sidebar to run a scenario.")


if __name__ == "__main__":
    main()