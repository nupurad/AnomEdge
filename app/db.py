import sqlite3
import os
import json
import time
import uuid
from typing import Any, Dict, Optional

DB_PATH = "data/edge_sentinel.db"  # change name if you want

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def reset_db():
    # Deletes the database file so everything is recreated fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def init_db(reset: bool = False):
    os.makedirs("data", exist_ok=True)

    if reset:
        reset_db()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,

        camera_id TEXT,
        zone TEXT,
        machine_id TEXT,

        anomaly_type TEXT NOT NULL,      -- normal|oil_leak|smoke_fire|belt_damage
        severity TEXT NOT NULL,          -- P0|P1|P2
        confidence REAL,                 -- 0..1

        summary TEXT NOT NULL,
        sop_refs_json TEXT,
        plan_json TEXT,

        image_path TEXT,
        model_name TEXT,
        connectivity TEXT,               -- offline|online

        status TEXT NOT NULL DEFAULT 'open',
        resolved_at INTEGER
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_events (
        id TEXT PRIMARY KEY,
        incident_id TEXT NOT NULL,
        timestamp INTEGER NOT NULL,

        event_type TEXT NOT NULL,        -- frame_captured|gemma_analyzed|sop_retrieved|tool_executed|sync_*
        data_json TEXT,

        FOREIGN KEY (incident_id) REFERENCES incidents(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS outbox (
        id TEXT PRIMARY KEY,
        incident_id TEXT NOT NULL,

        event_type TEXT NOT NULL,        -- incident_created|incident_resolved
        payload_json TEXT NOT NULL,

        status TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0,
        next_attempt_at INTEGER,
        last_error TEXT,

        created_at INTEGER NOT NULL,

        FOREIGN KEY (incident_id) REFERENCES incidents(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cache (
        key TEXT PRIMARY KEY,            -- e.g. sop::smoke
        value_json TEXT NOT NULL,
        kind TEXT NOT NULL,              -- sop|llm
        created_at INTEGER NOT NULL,
        ttl_seconds INTEGER,
        hits INTEGER NOT NULL DEFAULT 0
    );
    """)

    # Helpful indexes (optional but recommended)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents(created_at);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_anomaly_type ON incidents(anomaly_type);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_incident ON audit_events(incident_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_kind ON cache(kind);")

    conn.commit()
    conn.close()

# --------------------
# Utilities
# --------------------

def now_ts() -> int:
    return int(time.time())

def new_id() -> str:
    return uuid.uuid4().hex

def _json_or_none(obj: Optional[Dict[str, Any]]) -> Optional[str]:
    return None if obj is None else json.dumps(obj)

# --------------------
# Incidents
# --------------------

def insert_incident(
    *,
    camera_id: Optional[str] = None,
    zone: Optional[str] = None,
    machine_id: Optional[str] = None,
    anomaly_type: str,
    severity: str,
    confidence: Optional[float],
    summary: str,
    sop_refs: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
    image_path: Optional[str] = None,
    model_name: Optional[str] = None,
    connectivity: str = "offline",
) -> str:
    incident_id = new_id()
    ts = now_ts()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO incidents (
            id, created_at, updated_at,
            camera_id, zone, machine_id,
            anomaly_type, severity, confidence,
            summary, sop_refs_json, plan_json,
            image_path, model_name, connectivity,
            status, resolved_at
        ) VALUES (
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            'open', NULL
        );
        """,
        (
            incident_id, ts, ts,
            camera_id, zone, machine_id,
            anomaly_type, severity, confidence,
            summary, _json_or_none(sop_refs), _json_or_none(plan),
            image_path, model_name, connectivity,
        ),
    )
    conn.commit()
    conn.close()
    return incident_id


def update_incident_plan(
    incident_id: str,
    *,
    summary: Optional[str] = None,
    sop_refs: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
) -> None:
    ts = now_ts()
    sets = ["updated_at = ?"]
    params: list[Any] = [ts]

    if summary is not None:
        sets.append("summary = ?")
        params.append(summary)
    if sop_refs is not None:
        sets.append("sop_refs_json = ?")
        params.append(_json_or_none(sop_refs))
    if plan is not None:
        sets.append("plan_json = ?")
        params.append(_json_or_none(plan))

    params.append(incident_id)
    sql = f"UPDATE incidents SET {', '.join(sets)} WHERE id = ?;"

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    conn.close()


def update_incident_status(
    incident_id: str,
    *,
    status: str,
    resolved: bool = False,
) -> None:
    ts = now_ts()
    resolved_at = ts if resolved else None

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE incidents
        SET status = ?, updated_at = ?, resolved_at = ?
        WHERE id = ?;
        """,
        (status, ts, resolved_at, incident_id),
    )
    conn.commit()
    conn.close()


# --------------------
# Audit events
# --------------------

def add_audit_event(
    *,
    incident_id: str,
    event_type: str,
    data: Optional[Dict[str, Any]] = None,
    timestamp: Optional[int] = None,
) -> str:
    audit_id = new_id()
    ts = timestamp if timestamp is not None else now_ts()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO audit_events (id, incident_id, timestamp, event_type, data_json)
        VALUES (?, ?, ?, ?, ?);
        """,
        (audit_id, incident_id, ts, event_type, _json_or_none(data)),
    )
    conn.commit()
    conn.close()
    return audit_id


# --------------------
# Outbox
# --------------------

def enqueue_outbox(
    *,
    incident_id: str,
    event_type: str,
    payload: Dict[str, Any],
    next_attempt_at: Optional[int] = None,
) -> str:
    outbox_id = new_id()
    ts = now_ts()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO outbox (
            id, incident_id,
            event_type, payload_json,
            status, attempts, next_attempt_at, last_error,
            created_at
        ) VALUES (
            ?, ?,
            ?, ?,
            'pending', 0, ?, NULL,
            ?
        );
        """,
        (outbox_id, incident_id, event_type, json.dumps(payload), next_attempt_at, ts),
    )
    conn.commit()
    conn.close()
    return outbox_id


# --------------------
# Policy support queries
# --------------------

def count_p1_last_30_min(
    *,
    zone: Optional[str] = None,
    machine_id: Optional[str] = None,
) -> int:
    """
    Used for SOP-SAF-004: two P1 events within 30 minutes -> escalate to P0.
    """
    cutoff = now_ts() - 30 * 60

    where = ["created_at >= ?", "severity = 'P1'", "status NOT IN ('false_positive')"]
    params: list[Any] = [cutoff]

    if zone is not None:
        where.append("zone = ?")
        params.append(zone)
    if machine_id is not None:
        where.append("machine_id = ?")
        params.append(machine_id)

    sql = f"SELECT COUNT(1) AS c FROM incidents WHERE {' AND '.join(where)};"

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    conn.close()
    return int(row["c"] if row else 0)


def has_recurring_within_24h(
    *,
    anomaly_type: str,
    zone: Optional[str] = None,
    machine_id: Optional[str] = None,
) -> bool:
    """
    Used for SOP-SAF-004: recurring issue within 24 hours -> maintenance inspection required.
    Not necessarily a severity change.
    """
    cutoff = now_ts() - 24 * 60 * 60

    where = ["created_at >= ?", "anomaly_type = ?", "status NOT IN ('false_positive')"]
    params: list[Any] = [cutoff, anomaly_type]

    if zone is not None:
        where.append("zone = ?")
        params.append(zone)
    if machine_id is not None:
        where.append("machine_id = ?")
        params.append(machine_id)

    sql = f"SELECT 1 FROM incidents WHERE {' AND '.join(where)} LIMIT 1;"

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    conn.close()
    return row is not None
