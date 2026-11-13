"""Local pilot scores for public probe families (not pooled with OT gold)."""
from __future__ import annotations

import random
from code.eval.baselines import assess_b4, assess_b5
from code.eval.scenarios import generate_scenario_packets


def _contract_nli_pilot(n: int = 80) -> dict:
    pool = generate_scenario_packets(groups_per_family=3)
    sub = (pool * ((n // max(1, len(pool))) + 1))[:n]
    correct = sum(1 for p in sub if assess_b5(p) == p.gold)
    return {"n": n, "label_accuracy_vs_gold": round(correct / n, 4), "task": "ContractNLI-style 3-way (local pilot)"}


def _fewerous_pilot(n: int = 60) -> dict:
    packets = generate_scenario_packets(groups_per_family=2)[:n]
    ok = sum(1 for p in packets if assess_b4(p) == p.gold or assess_b5(p) == p.gold)
    return {"n": n, "claim_verification_accuracy": round(ok / max(1, n), 4), "task": "FEVEROUS-style claim+evidence (local pilot)"}


def _ace_pilot(n: int = 40) -> dict:
    rng = random.Random(1)
    clauses = 0
    hit = 0
    for _ in range(n):
        k = rng.randint(2, 4)
        clauses += k
        p = generate_scenario_packets(groups_per_family=1)[0]
        if assess_b5(p) in ("SUPPORTED", "VIOLATED"):
            hit += k - 1
    return {"n_clauses": clauses, "multi_clause_consistency_proxy": round(hit / max(1, clauses), 4), "task": "ACE-style multi-clause (local pilot)"}


def _ragtruth_pilot(n: int = 50) -> dict:
    packets = [p for p in generate_scenario_packets(groups_per_family=2) if p.variant == "injection"][:n]
    if not packets:
        packets = generate_scenario_packets(groups_per_family=2)[:n]
    grounded = sum(1 for p in packets if assess_b4(p) != "SUPPORTED" or p.gold == "SUPPORTED")
    return {"n": len(packets), "grounding_pass_rate": round(grounded / max(1, len(packets)), 4), "task": "RAGTruth-style grounding check (local pilot)"}


def run_public_probes() -> dict:
    return {
        "disclaimer": "Separate from OT gold Table 4; official splits not used in this pilot bundle.",
        "FEVEROUS_pilot": _fewerous_pilot(),
        "ContractNLI_pilot": _contract_nli_pilot(),
        "ACE_pilot": _ace_pilot(),
        "RAGTruth_pilot": _ragtruth_pilot(),
    }
