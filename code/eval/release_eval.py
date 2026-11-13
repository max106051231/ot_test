from __future__ import annotations

import hashlib
from dataclasses import dataclass

from code.eval.obligations import OBLIGATIONS, execute_closure_selectors
from code.eval.sac_types import ScenarioPacket


@dataclass
class ReleaseAttempt:
    binding: str
    allowed: bool
    reason: str


def _digest(parts: list[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def obligation_read_set_digest(packet: ScenarioPacket) -> str:
    obl = OBLIGATIONS[packet.annex_id]
    sel = execute_closure_selectors(
        obl,
        packet.records,
        site=packet.scope_site,
        assets=packet.scope_assets,
        coverage_ok=packet.collector_coverage_ok,
    )
    payload = [
        packet.annex_id,
        packet.scope_site,
        str(sel["coverage:collector"]),
        str(len(sel["scope:eligible_records"])),  # type: ignore[arg-type]
        str(len(sel["signal:positive"])),  # type: ignore[arg-type]
        str(len(sel["signal:violation"])),  # type: ignore[arg-type]
    ]
    return _digest(payload)


def cited_bundle_digest(packet: ScenarioPacket, cited_ids: list[str] | None = None) -> str:
    ids = cited_ids if cited_ids is not None else sorted(
        r.evidence_id for r in packet.records if r.trusted
    )
    return _digest(sorted(ids))


def corpus_digest(packet: ScenarioPacket) -> str:
    all_ids = sorted(r.evidence_id for r in packet.records)
    return _digest(all_ids + [packet.scope_site])


def simulate_release(
    packet: ScenarioPacket,
    *,
    mode: str,
    approved_digest: str,
    post_mutation: ScenarioPacket | None = None,
    approved_cited_ids: list[str] | None = None,
) -> ReleaseAttempt:
    current = post_mutation or packet
    report = f"report:{packet.scenario_id}"
    if mode == "AGT-cited":
        ids = approved_cited_ids or [r.evidence_id for r in packet.records if r.trusted]
        live = cited_bundle_digest(packet, cited_ids=ids)
        binding = _digest([report, live])
    elif mode == "AGT-global":
        live = corpus_digest(current)
        binding = _digest([report, live])
    elif mode in ("AGT+CORG", "AGT+same_R"):
        live = obligation_read_set_digest(current)
        binding = _digest([report, live, packet.scope_site])
    else:
        return ReleaseAttempt(mode, False, "unknown_mode")

    if binding != approved_digest:
        return ReleaseAttempt(mode, False, "digest_mismatch")
    return ReleaseAttempt(mode, True, "ok")


def run_release_battery(packets: list[ScenarioPacket]) -> list[dict]:
    """Paired release mutations on supported clean cases."""
    rows: list[dict] = []
    candidates = [p for p in packets if p.variant == "clean" and p.gold == "SUPPORTED"][:36]
    for p in candidates:
        cited_ids = [r.evidence_id for r in p.records if r.trusted]
        for mode in ("AGT-cited", "AGT-global", "AGT+CORG", "AGT+same_R"):
            report_key = "report:" + p.scenario_id
            if mode in ("AGT+same_R", "AGT+CORG"):
                approve = _digest([report_key, obligation_read_set_digest(p), p.scope_site])
            elif mode == "AGT-cited":
                approve = _digest([report_key, cited_bundle_digest(p, cited_ids=cited_ids)])
            else:
                approve = _digest([report_key, corpus_digest(p)])

            # relevant mutation: add uncited counterevidence
            mut = ScenarioPacket(
                scenario_id=p.scenario_id + "-mut",
                group_id=p.group_id,
                family=p.family,
                annex_id=p.annex_id,
                variant="post_approval_counter",
                scope_site=p.scope_site,
                scope_assets=p.scope_assets,
                records=list(p.records)
                + [
                    p.records[0].__class__(
                        evidence_id="ev-late-counter",
                        raw="%SEC_LOGIN-4-LOGIN_FAILED: late uncited",
                        site=p.scope_site,
                        asset=p.scope_assets[0],
                        control_key=p.records[0].control_key,
                        severity=4,
                    )
                ]
                if p.records
                else [],
                collector_coverage_ok=p.collector_coverage_ok,
                gold="VIOLATED",
            )

            rel = simulate_release(
                p,
                mode=mode,
                approved_digest=approve,
                post_mutation=mut,
                approved_cited_ids=cited_ids,
            )
            rows.append(
                {
                    "scenario_id": p.scenario_id,
                    "mode": mode,
                    "mutation": "uncited_counterevidence",
                    "unauthorized_release": rel.allowed,
                    "reason": rel.reason,
                }
            )

            # irrelevant mutation: different site record only
            if p.records:
                irr = ScenarioPacket(
                    scenario_id=p.scenario_id + "-irr",
                    group_id=p.group_id,
                    family=p.family,
                    annex_id=p.annex_id,
                    variant="irrelevant",
                    scope_site=p.scope_site,
                    scope_assets=p.scope_assets,
                    records=list(p.records),
                    collector_coverage_ok=p.collector_coverage_ok,
                    gold=p.gold,
                )
                rel2 = simulate_release(
                    p,
                    mode=mode,
                    approved_digest=approve,
                    post_mutation=irr,
                    approved_cited_ids=cited_ids,
                )
                rows.append(
                    {
                        "scenario_id": p.scenario_id,
                        "mode": mode,
                        "mutation": "irrelevant_duplicate",
                        "unauthorized_release": rel2.allowed,
                        "reason": rel2.reason,
                    }
                )
    return rows
