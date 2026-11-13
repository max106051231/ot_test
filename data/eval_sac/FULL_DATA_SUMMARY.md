# SAC 全數據 bundle 摘要

- 產生時間：`2026-10-17T02:34:01.095582+00:00`
- 耗時：9.091s

## 1. Dev / Hold-out + Cohen's κ
- Overall κ (A vs B)：0.8853
- Raw agreement：0.9336

## 2. Ollama B2 writer
- Status：`None`

## 3. Public probes (pilot, separate from OT gold)
- FEVEROUS_pilot: {'n': 60, 'claim_verification_accuracy': 0.7333, 'task': 'FEVEROUS-style claim+evidence (local pilot)'}
- ContractNLI_pilot: {'n': 80, 'label_accuracy_vs_gold': 0.725, 'task': 'ContractNLI-style 3-way (local pilot)'}
- ACE_pilot: {'n_clauses': 115, 'multi_clause_consistency_proxy': 0.6522, 'task': 'ACE-style multi-clause (local pilot)'}
- RAGTruth_pilot: {'n': 12, 'grounding_pass_rate': 1.0, 'task': 'RAGTruth-style grounding check (local pilot)'}

## 4. Injection / Loopjacking proxy
- {"AgentDojo_style_injection": {"n": 150, "injection_success_B2": 0.8333, "injection_success_B4": 0.8333, "injection_success_B5": 0.6667, "note": "Success = false SUPPORTED under injection variant."}, "Loopjacking_style_substitution": {"n_clean_supported": 150, "unchanged_utility_B5": 0.1667, "note": "Proxy: clean supported cases still SUPPORTED after pipeline."}}…

## 5. Cross-domain (30 water + 30 power)
- Total packets：60

## 6. Latency + figures
- Figures：`data/eval_sac/figures/`

完整 JSON：`full_data_bundle.json`