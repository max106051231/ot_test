from __future__ import annotations

from code.eval.obligations import OBLIGATIONS, corg_assess, execute_closure_selectors
from code.eval.sac_types import ScenarioPacket, Verdict


def _obl_for(packet: ScenarioPacket):
    return OBLIGATIONS[packet.annex_id]


def _draft_facts(packet: ScenarioPacket) -> dict:
    """Simulated auditor draft (optimistic)."""
    obl = _obl_for(packet)
    pos = any(r.control_key == obl.control_key for r in packet.records)
    return {"claims_positive": pos, "control_key": obl.control_key}


def assess_b0(packet: ScenarioPacket) -> Verdict:
    facts = _draft_facts(packet)
    if facts["claims_positive"]:
        return "SUPPORTED"
    return "INSUFFICIENT"


def assess_b0_full(packet: ScenarioPacket) -> Verdict:
    obl = _obl_for(packet)
    sel = execute_closure_selectors(
        obl,
        packet.records,
        site=packet.scope_site,
        assets=packet.scope_assets,
        coverage_ok=packet.collector_coverage_ok,
    )
    if sel["signal:violation"]:
        return "VIOLATED"
    if sel["signal:positive"] and sel["coverage:collector"]:
        return "SUPPORTED"
    return "INSUFFICIENT"


def _retrieve_bm25_like(packet: ScenarioPacket) -> list:
    obl = _obl_for(packet)
    tokens = obl.positive_signals[:2]
    out = []
    for r in packet.records:
        if not r.trusted:
            continue
        u = r.raw.upper()
        if any(t in u for t in tokens):
            out.append(r)
    if not out:
        out = [r for r in packet.records if r.trusted and r.control_key == obl.control_key]
    return out


def _graph_expand(records: list, packet: ScenarioPacket) -> list:
    """Same-site neighbors (typed graph stub)."""
    seen = {r.evidence_id for r in records}
    extra = []
    for r in packet.records:
        if r.evidence_id in seen:
            continue
        if r.site == packet.scope_site and r.trusted:
            extra.append(r)
    return records + extra[:2]


def assess_b2(packet: ScenarioPacket) -> Verdict:
    obl = _obl_for(packet)
    hits = _retrieve_bm25_like(packet)
    if any(r.severity <= 3 for r in hits):
        return "VIOLATED"
    if hits:
        return "SUPPORTED"
    return "INSUFFICIENT"


def assess_b3(packet: ScenarioPacket) -> Verdict:
    hits = _graph_expand(_retrieve_bm25_like(packet), packet)
    if any(r.severity <= 3 for r in hits):
        return "VIOLATED"
    if hits:
        return "SUPPORTED"
    return "INSUFFICIENT"


def assess_b4(packet: ScenarioPacket) -> Verdict:
    """B3 + generic grounding：有檢索命中與 span 長度，但不執行義務閉包／嚴格 scope。"""
    obl = _obl_for(packet)
    hits = _graph_expand(_retrieve_bm25_like(packet), packet)
    scoped = [r for r in hits if r.control_key == obl.control_key and len(r.raw) >= 8]
    if any(r.severity <= 3 for r in scoped):
        return "VIOLATED"
    if scoped:
        return "SUPPORTED"
    return "INSUFFICIENT"


def assess_b5(packet: ScenarioPacket) -> Verdict:
    obl = _obl_for(packet)
    v, _, _ = corg_assess(
        obl,
        packet.records,
        site=packet.scope_site,
        assets=packet.scope_assets,
        coverage_ok=packet.collector_coverage_ok,
        require_cited_span=True,
    )
    return v


BASELINES = {
    "B0": assess_b0,
    "B0-full": assess_b0_full,
    "B2": assess_b2,
    "B3": assess_b3,
    "B4": assess_b4,
    "B5": assess_b5,
}
