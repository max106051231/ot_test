from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Callable

from code.eval.sac_types import ScenarioPacket, Verdict


def _is_answerable(gold: Verdict) -> bool:
    return gold in ("SUPPORTED", "VIOLATED")


def compute_judgment_metrics(
    packets: list[ScenarioPacket],
    preds: list[Verdict],
) -> dict:
    assert len(packets) == len(preds)
    n = len(packets)
    gold_non_sup = [i for i, p in enumerate(packets) if p.gold != "SUPPORTED"]
    false_aff = sum(1 for i in gold_non_sup if preds[i] == "SUPPORTED")
    far = (false_aff / len(gold_non_sup)) if gold_non_sup else 0.0

    ans_idx = [i for i, p in enumerate(packets) if _is_answerable(p.gold)]
    correct_det = sum(1 for i in ans_idx if preds[i] == packets[i].gold)
    resolution = (correct_det / len(ans_idx)) if ans_idx else 0.0

    labels: list[Verdict] = ["SUPPORTED", "VIOLATED", "INSUFFICIENT"]
    f1s = []
    for lab in labels:
        tp = sum(1 for i, p in enumerate(packets) if p.gold == lab and preds[i] == lab)
        fp = sum(1 for i, p in enumerate(packets) if p.gold != lab and preds[i] == lab)
        fn = sum(1 for i, p in enumerate(packets) if p.gold == lab and preds[i] != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)

    return {
        "n": n,
        "FAR": round(far, 6),
        "resolution": round(resolution, 6),
        "macro_F1": round(macro_f1, 6),
        "false_affirmative_count": false_aff,
        "gold_non_supported": len(gold_non_sup),
        "answerable_n": len(ans_idx),
    }


def paired_bootstrap_ci(
    packets: list[ScenarioPacket],
    pred_a: list[Verdict],
    pred_b: list[Verdict],
    metric_fn: Callable[[list[ScenarioPacket], list[Verdict]], float],
    *,
    n_draws: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Returns (observed_delta_a_minus_b, ci_low, ci_high) on metric (e.g. FAR)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(packets):
        groups[p.group_id].append(i)

    def metric_on(idxs: list[int], preds: list[Verdict]) -> float:
        sub_p = [packets[i] for i in idxs]
        sub_v = [preds[i] for i in idxs]
        return metric_fn(sub_p, sub_v)

    def far_only(ps, pr):
        m = compute_judgment_metrics(ps, pr)
        return m["FAR"]

    observed = metric_on(list(range(len(packets))), pred_a) - metric_on(list(range(len(packets))), pred_b)
    rng = random.Random(seed)
    gids = list(groups.keys())
    deltas = []
    for _ in range(n_draws):
        sample_g = [rng.choice(gids) for _ in range(len(gids))]
        idxs = []
        for g in sample_g:
            idxs.extend(groups[g])
        deltas.append(metric_on(idxs, pred_a) - metric_on(idxs, pred_b))
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    return round(observed, 6), round(lo, 6), round(hi, 6)


def summarize_release(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for mode in ("AGT-cited", "AGT-global", "AGT+CORG", "AGT+same_R"):
        rel = [r for r in rows if r["mode"] == mode and r["mutation"] == "uncited_counterevidence"]
        irr = [r for r in rows if r["mode"] == mode and r["mutation"] == "irrelevant_duplicate"]
        n_rel = len(rel) or 1
        n_irr = len(irr) or 1
        out[mode] = {
            "unauthorized_release_rate": round(
                sum(1 for r in rel if r["unauthorized_release"]) / n_rel, 6
            ),
            "irrelevant_still_release_rate": round(
                sum(1 for r in irr if r["unauthorized_release"]) / n_irr, 6
            ),
            "n_relevant_trials": len(rel),
            "n_irrelevant_trials": len(irr),
        }
    return out
