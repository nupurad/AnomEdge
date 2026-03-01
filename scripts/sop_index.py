from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class SOP:
    sop_id: str
    title: str
    triggers: List[str]
    severity_guidance: str
    immediate_actions: List[str]
    escalation_criteria: List[str]
    required_logging: List[str]
    raw_text: str


# -----------------------------
# Parsing
# -----------------------------

def _extract_section(md: str, heading: str) -> str:
    lines = md.splitlines()
    out = []
    in_section = False
    for line in lines:
        if re.match(rf"^###\s+{re.escape(heading)}\s*$", line.strip()):
            in_section = True
            continue
        if in_section and re.match(r"^###\s+", line.strip()):
            break
        if in_section:
            out.append(line)
    return "\n".join(out).strip()


def _extract_title(md: str, fallback: str) -> str:
    for ln in md.splitlines():
        if ln.startswith("# "):
            return ln[2:].strip()
    return fallback


def _extract_list(section_text: str) -> List[str]:
    items = []
    for ln in section_text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"^[-*]\s+(.*)$", ln)
        if m:
            items.append(m.group(1))
            continue
        m = re.match(r"^\d+\.\s+(.*)$", ln)
        if m:
            items.append(m.group(1))
    return items


def parse_sop(path: str) -> SOP:
    raw = open(path, "r", encoding="utf-8").read()
    sop_id = os.path.splitext(os.path.basename(path))[0]
    title = _extract_title(raw, sop_id)

    return SOP(
        sop_id=sop_id,
        title=title,
        triggers=_extract_list(_extract_section(raw, "Trigger Conditions")),
        severity_guidance=_extract_section(raw, "Severity Guidance"),
        immediate_actions=_extract_list(_extract_section(raw, "Immediate Actions")),
        escalation_criteria=_extract_list(_extract_section(raw, "Escalation Criteria")),
        required_logging=_extract_list(_extract_section(raw, "Required Logging")),
        raw_text=raw,
    )


def load_sops(directory: str = "data/sop") -> List[SOP]:
    paths = sorted(glob.glob(os.path.join(directory, "*.md")))
    return [parse_sop(p) for p in paths]


# -----------------------------
# Retrieval
# -----------------------------

def _score(sop: SOP, query_terms: List[str]) -> int:
    blob = (
        sop.sop_id
        + sop.title
        + " ".join(sop.triggers)
        + sop.severity_guidance
        + " ".join(sop.immediate_actions)
    ).lower()

    return sum(1 for t in query_terms if t and t.lower() in blob)


def retrieve_sops(
    sops: List[SOP],
    anomaly_type: str,
    severity: Optional[str],
    observations: List[str],
    top_k: int = 2,
) -> List[SOP]:
    query = [anomaly_type, severity or ""] + observations
    ranked: List[Tuple[int, SOP]] = [(_score(s, query), s) for s in sops]
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in ranked[:top_k]]

def _is_safety_framework(sop: SOP) -> bool:
    return sop.sop_id.upper().startswith("SOP-SAF-004")

def retrieve_sops_with_framework(
    sops: List[SOP],
    anomaly_type: str,
    severity: Optional[str],
    observations: List[str],
    top_k: int = 2,
) -> List[SOP]:
    # normal retrieval
    matches = retrieve_sops(sops, anomaly_type, severity, observations, top_k=top_k)

    # append SAF framework if present and not already included
    framework = next((s for s in sops if _is_safety_framework(s)), None)
    if framework and all(m.sop_id != framework.sop_id for m in matches):
        matches.append(framework)

    return matches