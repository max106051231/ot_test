from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["SUPPORTED", "VIOLATED", "INSUFFICIENT"]


@dataclass
class EvidenceRecord:
    evidence_id: str
    raw: str
    site: str
    asset: str
    control_key: str
    severity: int
    ingest_ok: bool = True
    trusted: bool = True


@dataclass
class ScenarioPacket:
    scenario_id: str
    group_id: str
    family: str
    annex_id: str
    variant: str
    scope_site: str
    scope_assets: list[str]
    records: list[EvidenceRecord]
    collector_coverage_ok: bool
    gold: Verdict
    requires_absence_proof: bool = False
    injection_goal: str = ""
    draft_positive: bool = True
