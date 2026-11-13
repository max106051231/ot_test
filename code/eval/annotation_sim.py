"""Development split + dual-annotator simulation and Cohen's kappa."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from code.eval.sac_types import ScenarioPacket, Verdict


@dataclass
class AnnotationRecord:
    scenario_id: str
    group_id: str
    split: str
    gold: Verdict
    annotator_a: Verdict
    annotator_b: Verdict


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    cats = sorted(set(labels_a) | set(labels_b))
    conf = Counter(zip(labels_a, labels_b))
    po = sum(conf.get((c, c), 0) for c in cats) / n
    pa = Counter(labels_a)
    pb = Counter(labels_b)
    pe = sum((pa[c] / n) * (pb[c] / n) for c in cats)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def assign_splits(packets: list[ScenarioPacket], *, dev_groups_per_family: int = 5) -> list[ScenarioPacket]:
    """First N groups per family → development; remainder → held_out."""
    seen: dict[str, int] = {}
    for p in packets:
        fam = p.family
        gnum = seen.get(fam, 0)
        if gnum == 0:
            # parse from group_id management_access-g003
            try:
                gnum = int(p.group_id.rsplit("-g", 1)[-1])
            except ValueError:
                gnum = 0
        split = "development" if gnum < dev_groups_per_family else "held_out"
        p.__dict__["eval_split"] = split  # type: ignore[attr-defined]
    return packets


def split_for_group_id(group_id: str, dev_groups_per_family: int = 5) -> str:
    try:
        gnum = int(group_id.rsplit("-g", 1)[-1])
    except ValueError:
        return "held_out"
    return "development" if gnum < dev_groups_per_family else "held_out"


def dual_annotate_packets(
    packets: list[ScenarioPacket],
    *,
    dev_groups_per_family: int = 5,
    disagree_rate: float = 0.07,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    records: list[AnnotationRecord] = []
    for p in packets:
        split = split_for_group_id(p.group_id, dev_groups_per_family)
        gold = p.gold
        ann_a = gold
        ann_b = gold
        if rng.random() < disagree_rate:
            pool: list[Verdict] = ["SUPPORTED", "VIOLATED", "INSUFFICIENT"]
            alts = [x for x in pool if x != gold]
            ann_b = rng.choice(alts)
        records.append(
            AnnotationRecord(
                scenario_id=p.scenario_id,
                group_id=p.group_id,
                split=split,
                gold=gold,
                annotator_a=ann_a,
                annotator_b=ann_b,
            )
        )

    dev = [r for r in records if r.split == "development"]
    hold = [r for r in records if r.split == "held_out"]
    all_a = [r.annotator_a for r in records]
    all_b = [r.annotator_b for r in records]
    gold_l = [r.gold for r in records]
    dev_a = [r.annotator_a for r in dev]
    dev_b = [r.annotator_b for r in dev]
    hold_a = [r.annotator_a for r in hold]
    hold_b = [r.annotator_b for r in hold]

    dev_groups = len({r.group_id for r in dev})
    hold_groups = len({r.group_id for r in hold})

    return {
        "method": "simulated_dual_annotation",
        "note": "Annotator A = gold rubric; B = gold + 7% label noise for pilot kappa (replace with expert review).",
        "dev_groups_per_family": dev_groups_per_family,
        "development": {
            "groups": dev_groups,
            "packets": len(dev),
            "cohen_kappa_a_vs_b": round(_cohen_kappa(dev_a, dev_b), 4),
        },
        "held_out": {
            "groups": hold_groups,
            "packets": len(hold),
            "cohen_kappa_a_vs_b": round(_cohen_kappa(hold_a, hold_b), 4),
        },
        "overall": {
            "packets": len(records),
            "cohen_kappa_a_vs_b": round(_cohen_kappa(all_a, all_b), 4),
            "cohen_kappa_gold_vs_b": round(_cohen_kappa(gold_l, all_b), 4),
            "raw_agreement_a_b": round(sum(1 for r in records if r.annotator_a == r.annotator_b) / len(records), 4),
        },
        "records_sample": [
            {
                "scenario_id": r.scenario_id,
                "split": r.split,
                "gold": r.gold,
                "annotator_a": r.annotator_a,
                "annotator_b": r.annotator_b,
            }
            for r in records[:12]
        ],
    }
