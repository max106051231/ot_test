"""Ollama-backed writer for B2-style three-way judgment (subset)."""
from __future__ import annotations

import json
import re
import time

from code.eval.baselines import assess_b2
from code.eval.obligations import OBLIGATIONS
from code.eval.sac_types import ScenarioPacket, Verdict


def _parse_verdict(text: str) -> Verdict | None:
    u = (text or "").upper()
    for v in ("INSUFFICIENT", "VIOLATED", "SUPPORTED"):
        if v in u:
            return v  # type: ignore[return-value]
    return None


def _prompt_for(packet: ScenarioPacket) -> list[dict]:
    obl = OBLIGATIONS[packet.annex_id]
    lines = []
    for r in packet.records[:8]:
        if r.trusted:
            lines.append(f"- [{r.evidence_id}] {r.raw[:200]}")
    body = "\n".join(lines) or "(no trusted records)"
    sys_msg = (
        "You are an OT compliance assessor. Reply with exactly one word: "
        "SUPPORTED, VIOLATED, or INSUFFICIENT. No explanation."
    )
    user = (
        f"Obligation: {packet.annex_id} / {obl.control_key}\n"
        f"Scope site={packet.scope_site} assets={packet.scope_assets}\n"
        f"Coverage OK={packet.collector_coverage_ok}\n"
        f"Evidence:\n{body}"
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def run_ollama_b2_subset(
    packets: list[ScenarioPacket],
    *,
    max_cases: int = 40,
    timeout_budget_s: float = 180.0,
) -> dict:
    from code.services import ollama_service

    t0 = time.perf_counter()
    if not ollama_service.is_ready():
        return {
            "status": "skipped",
            "reason": ollama_service.current_info().get("ollama_error") or "Ollama not ready",
            "fallback": "deterministic B2 proxy on full set",
            "deterministic_b2": _deterministic_metrics(packets),
        }

    subset = [p for p in packets if p.variant in ("clean", "evidence_removed", "scope_mismatch")][:max_cases]
    preds: list[Verdict] = []
    gold: list[Verdict] = []
    lat_ms: list[float] = []

    for p in subset:
        if time.perf_counter() - t0 > timeout_budget_s:
            break
        msgs = _prompt_for(p)
        t1 = time.perf_counter()
        try:
            reply = ollama_service.chat(msgs, max_new_tokens=32, allow_fallback=False)
        except Exception as ex:
            reply = f"INSUFFICIENT {ex}"
        lat_ms.append((time.perf_counter() - t1) * 1000)
        v = _parse_verdict(reply) or "INSUFFICIENT"
        preds.append(v)
        gold.append(p.gold)

    from code.eval.metrics import compute_judgment_metrics

    sub = subset[: len(preds)]
    m = compute_judgment_metrics(sub, preds)

    det = [assess_b2(p) for p in sub]
    m_det = compute_judgment_metrics(sub, det)

    lat_ms.sort()
    p50 = lat_ms[len(lat_ms) // 2] if lat_ms else 0
    p95 = lat_ms[int(len(lat_ms) * 0.95)] if lat_ms else 0

    return {
        "status": "ok",
        "model": ollama_service.current_model_name(),
        "n_cases": len(preds),
        "ollama_b2_metrics": m,
        "deterministic_b2_same_subset": m_det,
        "latency_ms": {"p50": round(p50, 2), "p95": round(p95, 2), "n": len(lat_ms)},
    }


def _deterministic_metrics(packets: list[ScenarioPacket]) -> dict:
    from code.eval.metrics import compute_judgment_metrics

    preds = [assess_b2(p) for p in packets]
    return compute_judgment_metrics(packets, preds)
