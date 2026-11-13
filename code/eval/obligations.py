from __future__ import annotations

from dataclasses import dataclass

from code.eval.sac_types import EvidenceRecord, Verdict


@dataclass
class ObligationTemplate:
    annex_id: str
    family: str
    control_key: str
    positive_signals: tuple[str, ...]
    violation_signals: tuple[str, ...]
    coverage_required: bool = True


OBLIGATIONS: dict[str, ObligationTemplate] = {
    "A.5.15": ObligationTemplate(
        "A.5.15", "management_access", "access_control",
        ("SEC_LOGIN", "LOGIN", "AAA", "RADIUS", "TACACS", "SSH"),
        ("LOGIN_FAILED", "AUTHFAIL", "denied"),
    ),
    "A.5.19": ObligationTemplate(
        "A.5.19", "supplier_context", "supplier_security",
        ("CDP", "LLDP", "NEIGHBOR", "REMOTE"),
        ("UNAUTHORIZED_NEIGHBOR", "CDP-4"),
    ),
    "A.8.7": ObligationTemplate(
        "A.8.7", "malware_protection", "malware_defense",
        ("IPS", "MALWARE", "THREAT", "ATTACK"),
        ("MALWARE", "EXPLOIT", "OUTBREAK"),
    ),
    "A.8.8": ObligationTemplate(
        "A.8.8", "patch_handling", "patch_management",
        ("IMAGE", "SOFTWARE", "UPGRADE", "SMU", "IOS"),
        ("VULN", "CVE", "UNSUPPORTED"),
    ),
    "A.8.19": ObligationTemplate(
        "A.8.19", "configuration_change", "recipe_audit",
        ("CONFIG_I", "INTERFACE", "PORT-SECURITY", "CHANGED"),
        ("CONFIG_I", "UNAUTHORIZED"),
    ),
    "A.8.24": ObligationTemplate(
        "A.8.24", "protected_traffic", "sec_gem_log",
        ("SNMP", "CRYPTO", "CERTIFICATE", "TLS", "PKI"),
        ("WEAK_CIPHER", "CERT-EXPIRED", "SNMP-4"),
    ),
}


def selector_queries(obl: ObligationTemplate) -> list[str]:
    return [
        f"scope:eligible_records:{obl.control_key}",
        f"signal:positive:{obl.control_key}",
        f"signal:violation:{obl.control_key}",
        f"coverage:collector:{obl.control_key}",
    ]


def _in_scope(rec: EvidenceRecord, site: str, assets: list[str]) -> bool:
    if rec.site != site:
        return False
    if assets and rec.asset not in assets:
        return False
    return True


def _match_any(raw: str, tokens: tuple[str, ...]) -> bool:
    u = (raw or "").upper()
    return any(t in u for t in tokens)


def execute_closure_selectors(
    obl: ObligationTemplate,
    records: list[EvidenceRecord],
    *,
    site: str,
    assets: list[str],
    coverage_ok: bool,
) -> dict[str, list[EvidenceRecord] | bool]:
    eligible = [r for r in records if r.trusted and _in_scope(r, site, assets)]
    pos = [r for r in eligible if r.control_key == obl.control_key and _match_any(r.raw, obl.positive_signals)]
    viol = [
        r for r in eligible
        if r.control_key == obl.control_key
        and (r.severity <= 3 or _match_any(r.raw, obl.violation_signals))
    ]
    return {
        "scope:eligible_records": eligible,
        "signal:positive": pos,
        "signal:violation": viol,
        "coverage:collector": coverage_ok,
    }


def corg_assess(
    obl: ObligationTemplate,
    records: list[EvidenceRecord],
    *,
    site: str,
    assets: list[str],
    coverage_ok: bool,
    require_cited_span: bool,
) -> tuple[Verdict, set[str], dict]:
    sel = execute_closure_selectors(obl, records, site=site, assets=assets, coverage_ok=coverage_ok)
    deps: set[str] = set(selector_queries(obl))
    if obl.coverage_required and not coverage_ok:
        return "INSUFFICIENT", deps, sel

    viol = sel["signal:violation"]  # type: ignore
    pos = sel["signal:positive"]  # type: ignore
    if viol:
        if require_cited_span and viol:
            return "VIOLATED", deps, sel
        return "VIOLATED", deps, sel
    if pos:
        if require_cited_span:
            cited = [r for r in pos if len((r.raw or "").strip()) >= 12]
            if not cited:
                return "INSUFFICIENT", deps, sel
        return "SUPPORTED", deps, sel
    return "INSUFFICIENT", deps, sel
