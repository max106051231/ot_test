"""30-group water / power OT cross-domain pilot (synthetic templates)."""
from __future__ import annotations

import random

from code.eval.baselines import assess_b4, assess_b5
from code.eval.log_bucket import make_record
from code.eval.sac_types import ScenarioPacket


def _domain_packet(domain: str, idx: int) -> ScenarioPacket:
    rng = random.Random(idx + (1 if domain == "water" else 99))
    site = f"{domain.upper()}-Plant-{idx % 5 + 1}"
    asset = f"PLC-{domain[0].upper()}{idx:02d}"
    if domain == "water":
        raw = f"SCADA {site} CHLORINE_SETPOINT_CHANGE seq={idx} operator=maintenance"
        key = "recipe_audit"
    else:
        raw = f"SUBSTATION {site} BREAKER_TRIP relay=87B idx={idx}"
        key = "access_control"
    sev = 4 if idx % 7 == 0 else 5
    rec = make_record(idx, key, raw, sev, site=site, asset=asset)
    gold = "VIOLATED" if sev <= 4 else "SUPPORTED"
    if idx % 5 == 3:
        gold = "INSUFFICIENT"
        recs = []
    else:
        recs = [rec]
    return ScenarioPacket(
        scenario_id=f"{domain}-cross-g{idx:03d}-clean",
        group_id=f"{domain}_cross-g{idx:03d}",
        family=f"{domain}_ot",
        annex_id="A.8.19" if domain == "water" else "A.5.15",
        variant="clean",
        scope_site=site,
        scope_assets=[asset],
        records=recs,
        collector_coverage_ok=True,
        gold=gold,  # type: ignore[arg-type]
    )


def run_cross_domain_eval(*, groups_per_domain: int = 30) -> dict:
    water = [_domain_packet("water", i) for i in range(groups_per_domain)]
    power = [_domain_packet("power", i) for i in range(groups_per_domain)]
    all_p = water + power

    def metrics(sub):
        from code.eval.metrics import compute_judgment_metrics

        b4 = [assess_b4(p) for p in sub]
        b5 = [assess_b5(p) for p in sub]
        return {
            "B4": compute_judgment_metrics(sub, b4),
            "B5": compute_judgment_metrics(sub, b5),
        }

    return {
        "groups_per_domain": groups_per_domain,
        "total_packets": len(all_p),
        "water": metrics(water),
        "power": metrics(power),
        "pooled": metrics(all_p),
        "note": "Synthetic cross-domain templates; not Fab-A/B cross-site.",
    }
