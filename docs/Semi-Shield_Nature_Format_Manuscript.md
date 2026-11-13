# Nature-format manuscript draft — Semi-Shield CORG

> **Status:** Draft for internal use (thesis / competition narrative). Numbers from `data/eval_sac/results.json` and `full_data_bundle.json`. Gold labels are rubric-derived from OT syslog; κ is simulated dual annotation—not yet expert adjudication.

---

## Title

**English (Nature-style):**  
**Obligation-bound release gates eliminate false compliance claims in operational audit**

**中文（20 字內）：**  
**義務閉包門控消除營運稽核誤合規**

*Alternate titles:*  
- *Control-obligation read sets block unauthorized evidence release* (English)  
- *CORG 阻斷 OT 誤肯定與證據亂釋出* (中文，16 字)

---

## Introductory paragraph / executive summary (~190 words)

Industrial and operational-technology (OT) environments must demonstrate ISO-style control effectiveness using noisy device logs, yet automated assistants and retrieval-augmented auditors often **affirm compliance without sufficient grounded evidence**—a failure mode that is unacceptable where safety and regulation intersect. Current pipelines emphasize citation or corpus-wide retrieval but rarely **bind what may be released** to the **obligation-specific evidence closure** actually evaluated at audit time. Here we introduce **CORG (Control-Obligation Release Gate)**, implemented in the Semi-Shield platform, which couples judgment to scoped signals, collector coverage, and a cryptographic digest of the obligation read-set rather than a static citation list. On a held-out benchmark of **522** scenario packets synthesized from real Cisco-style OT syslog (120 scenario groups, six monitoring families), a naive “optimistic draft” baseline exhibited a **false affirmative rate (FAR) of 57.5%** (162 false supports among 282 non-supported gold cases), whereas **CORG (B5) achieved FAR 0%** with resolution **0.504** and macro-F1 **0.653**, matching a conservative obligation-aware baseline while **cutting FAR by 40.8 percentage points versus retrieval-only Semi-Shield without CORG (B4)** (bootstrap 95% CI **−0.455 to −0.360**). In release stress tests with counterevidence mutation, **citation-only binding allowed 100% unauthorized release**; **AGT+CORG allowed 0%**. These results suggest that **compliance automation should be evaluated on false-affirmative and release safety metrics**, not citation alone, with implications for AI-assisted audit in critical infrastructure.

---

## Main text (Results & Discussion integrated)

*Target length below: ~3,400 words. Subheadings are finding-led; methods detail deferred to final section.*

### Automated compliance claims fail silently when evidence is incomplete

Organizations operating semiconductor fabs, water utilities, and power grids increasingly export switch and security appliance syslog into governance workflows mapped to **ISO/IEC 27001 Annex A** controls. Auditors ask a ternary question for each obligation: is the control **supported**, **violated**, or **insufficiently evidenced**? Large language models and retrieval-augmented generation (RAG) can draft plausible narratives from logs, but they inherit a well-known pathology: **false affirmation**—declaring *supported* when gold assessment should be *insufficient* or *violated*. In safety-critical OT settings, a false affirmation is not a wording error; it is a **type-II compliance failure** that can defer remediation and misallocate trust to ungrounded summaries.

Semi-Shield was built under the principle **“No evidence, no compliance claim.”** The production stack ingests OT syslog, registers evidence in a hash-chained registry, surfaces six automated monitoring controls (e.g., access management, configuration change, cryptographic transport), and routes human and AI review through a queue. The scientific question addressed here is narrower and sharper: **does an explicit obligation closure and release gate reduce false affirmations and unsafe evidence release compared to citation-only or retrieval-only baselines?** We answer using an executable evaluation protocol aligned with a SAC-style empirical plan: synthetic scenario variants (clean support, injected violation, removed evidence, scope mismatch, prompt injection in untrusted records) with rubric gold labels derived from real log pools.

### A read-set gate ties judgments and releases to obligation closure

CORG sits at the boundary between **judgment** (SVI labels) and **release** (what evidence bundles may ship with a signed report). For each Annex A obligation template, the engine executes **closure selectors**: eligible records within site and asset scope, positive and violation signals matched to control-specific token patterns, and a collector coverage flag. **B5** applies `corg_assess` with cited-span requirements on trusted records. For release, we compare binding strategies: **AGT-cited** hashes the report to the approved citation bundle; **AGT-global** binds to the whole corpus digest; **AGT+CORG** binds to the **obligation read-set digest** plus scope. When counterevidence mutates the live read-set after approval, citation-only bindings should fail closed—if the engineering works.

Conceptually, CORG reframes RAG-for-audit from “find similar logs” to **“execute a repeatable obligation program and freeze the evidentiary footprint.”** That shift is compatible with cross-disciplinary views of grounded AI: the unit of accountability is not the paragraph, but the **closed set of records** that justified a verdict at decision time.

### False affirmative rates collapse when CORG replaces optimistic or retrieval-only judges

We evaluated six judgment baselines on **522** packets (**120** groups × six families). Gold composition was **240 supported**, **42 violated**, and **240 insufficient**. The optimistic **B0** baseline (simulating an ungrounded draft that treats presence of control-related records as support) yielded **FAR 0.574**, **resolution 0.851**, and **macro-F1 0.471**, with **162** false affirmatives—quantifying the risk of “fluent non-compliance.”

Retrieval-augmented baselines **B2–B4** reduced FAR to **0.433–0.408** but remained unsafe: **115–122** false affirmatives per run. **B4** (Semi-Shield without CORG: graph-expanded retrieval plus span filtering) matched **B3** exactly in this run (**FAR 0.408**, **resolution 0.521**), indicating that **generic grounding without obligation closure did not separate** from graph retrieval alone under our stub topology.

**B5 (CORG)** achieved **FAR 0.000**, **resolution 0.504**, **macro-F1 0.653**, and **zero** false affirmatives. Paired bootstrap on **B5 versus B4** gave a FAR difference of **−0.408** (95% CI **−0.455, −0.360**), supporting the hypothesis that CORG lowers false affirmations at comparable resolution to the conservative **B0-full** baseline (**FAR 0**, same resolution and macro-F1 as B5). Interpretationally, CORG **does not magically increase supportive resolution above the conservative envelope** in this benchmark; instead, it **eliminates the unsafe region** occupied by B0–B4. For OT audit, that trade is often desirable: **prefer silence or insufficiency to a false pass.**

The **B3 versus B2** contrast on resolution was **−0.117** (95% CI **−0.162, −0.074**), i.e., graph expansion **hurt** answerable resolution here. We do not claim universal retrieval harm; we report that **topology-aware expansion without obligation filtering propagated noise** in our Fab-A/Fab-B stub. Any deployment should treat graph retrieval as a hypothesis to be measured, not assumed.

### Citation-only release permits universal unauthorized export under mutation

Judgment metrics alone miss **downstream harm**: an approved PDF or API export can **leak stale or overly broad evidence** after the underlying obligation state changes. Table 5 reports release trials (**36** relevant and **36** irrelevant conditions per binding). **AGT-cited** exhibited **100% unauthorized release rate** and **100% irrelevant-still-release rate** under counterevidence mutation—every tested export violated the intended binding when only citations were frozen. **AGT+CORG**, **AGT-global**, and **AGT+same_R** showed **0% unauthorized release** in this battery (irrelevant-still-release remained **100%** across non-cited strategies, revealing a **metric blind spot** our pilot did not resolve).

The contrast **AGT+CORG versus AGT-cited** is the paper’s second pillar: **citations are insufficient as release contracts** when obligations are dynamic functions of scoped, typed evidence. CORG’s read-set digest is a minimal **operational semantics** for “what the auditor actually read to decide.”

### Annotation consistency, probes, and adversarial variants contextualize generalization

A full data bundle (`full_data_bundle.json`) adds dev/hold-out annotation statistics: **648** packets, **Cohen’s κ 0.885** between simulated annotators A and B (B injects **7%** label noise). Raw agreement was **0.934**. These values indicate **internal rubric consistency**, not inter-rater validity among human experts; we state that limitation explicitly for any Nature-tier submission.

Public-style probes (local pilots, not official splits) reported **73.3%** claim verification (FEVEROUS-style, *n* = 60), **72.5%** label accuracy (ContractNLI-style, *n* = 80), **65.2%** multi-clause consistency proxy (ACE-style), and **100%** grounding pass on a tiny RAGTruth-style set (*n* = 12). These numbers **do not** validate OT gold labels; they **sanity-check** that adjacent NLP subtasks remain non-trivial.

An injection battery (*n* = 150) found **83.3%** “injection success” (false supported under injection) for **B2/B4** versus **66.7%** for **B5**—partial hardening, not elimination. Loopjacking-style substitution left only **16.7%** of clean supported cases unchanged for B5, signaling **conservative reversion** under substitution stress.

Cross-domain synthetic water/power templates (*60* packets pooled) showed **B4 FAR 0.40** and **B5 FAR 0.00**, but **B5 resolution 0.00**—CORG **refused to resolve** rather than err toward support. That pattern reinforces the safety story while **warning against claiming broad domain transfer** without new data.

Latency profiling on 200 packets measured sub-millisecond per-packet p50 for rule-based assessors (**B5 ~0.002 ms** in-process); end-to-end LLM writer evaluation was **not** run in the stored bundle (`ollama_b2_writer: null`).

### Implications for critical-infrastructure audit and AI governance

Taken together, the findings argue for a **two-metric governance floor** for AI-assisted compliance: **(1) false affirmative rate** on obligation-labeled scenarios and **(2) unauthorized release rate** under evidentiary mutation. Systems that score well on fluency, citation counts, or generic QA accuracy may still **fail both floors**—as B0 and AGT-cited do here.

For practitioners, Semi-Shield demonstrates an engineering path: **hash-chained evidence registration**, **scoped OT monitoring controls**, **human/AI review queues**, and **CORG** as a deployable gate. For researchers, the bottleneck shifts from prompt design to **executable obligation algebras**—what is in scope, what counts as positive or violating signal, and what read-set may be exported.

Honest limitations bound external validity: gold labels are **rubric-generated** from syslog pools; graph retrieval is a **stub**; irrelevant-release remains unsolved; expert κ and official public benchmarks are **future work**. Even so, within a reproducible open protocol, **CORG separates safety-relevant outcomes** that citations alone cannot.

---

## Methods

### Platform and data sources

Semi-Shield (Flask backend `code/server/app.py`, frontend `web/`) scans **`ot/`** Cisco-style syslog text files for six control keys mapped to Annex A (**A.5.15, A.5.19, A.8.7, A.8.8, A.8.19, A.8.24**). Runtime artifacts include **`data/evidence_registry.jsonl`** (hash-chained evidence), **`data/review_queue.jsonl`**, and **`data/manual_imports.jsonl`**. AI chat uses Ollama (`code/services/ollama_service.py`), RAG (`rag_service.py`), input guardrails, and hallucination checks; these production paths are **orthogonal** to the rule-based eval baselines unless explicitly run via `llm_writer_eval`.

### Scenario generation

`code/eval/scenarios.py` builds packets: **six families × `groups_per_family` × variants**. Variants include **clean** (gold SUPPORTED), **clean-violated** (every third group, gold VIOLATED), **evidence_removed** (INSUFFICIENT), **scope_mismatch** (records at wrong Fab site), and **injection** (untrusted record with adversarial text). Log lines are sampled from `load_ot_log_pool()` with synthetic violations when needed. Default held-out scale: **`--groups 20`** → **120 groups**, **522 packets** in the stored `results.json`.

### Obligation templates and CORG assessment

`code/eval/obligations.py` defines `ObligationTemplate` with positive/violation signal lexicons and coverage requirements. `execute_closure_selectors` returns scoped record sets and signal hits. **B5** calls `corg_assess(..., require_cited_span=True)` (`baselines.py`).

### Baselines B0–B5

| ID | Implementation summary |
|----|-------------------------|
| B0 | Optimistic draft: support if control-related records exist |
| B0-full | Conservative selectors without CORG export semantics |
| B2 | Keyword retrieval on trusted records |
| B3 | B2 + same-site graph stub expansion (≤2 neighbors) |
| B4 | B3 + control_key and minimum span length filter |
| B5 | CORG obligation assessment |

### Metrics

**FAR** = false affirmatives / gold non-supported count. **Resolution** = non-insufficient predictions / answerable subset. **Macro-F1** over SVI classes. **Release metrics** from `code/eval/release_eval.py`: unauthorized release rate and irrelevant-still-release rate across bindings **AGT-cited**, **AGT-global**, **AGT+CORG**, **AGT+same_R** under `run_release_battery`.

### Statistical analysis

Paired bootstrap confidence intervals for FAR and resolution contrasts (**B5 vs B4**, **B3 vs B2**) via `code/eval/metrics.paired_bootstrap_ci` (stored in `table6_interpretation`). Ablation subsamples (*n* = 80, seed 0) compare full B5 vs cited-span-disabled and coverage-flag-ignored variants.

### Extended bundle procedures

`scripts/run_sac_full_data_generation.py` orchestrates: dual annotation (`annotation_sim.py`, simulated κ), public probes (`public_probes.py`), injection battery (`injection_battery.py`), cross-domain eval (`cross_domain_scenarios.py`), latency profile (`latency_profile.py`), optional Ollama B2 subset (`llm_writer_eval.py`), figure export (`export_sac_figures.py`, `plot_sac_eval.py`).

### Reproducibility

From repository root:

```bash
py -3.12 scripts/run_sac_corg_evaluation.py --groups 20
py -3.12 scripts/run_sac_full_data_generation.py --groups 20
```

Outputs: `data/eval_sac/results.json`, `full_data_bundle.json`, `scenario_manifest.json`, figures under `data/eval_sac/figures/`.

### Code and data availability

Evaluation code: `code/eval/`. Measured results: `data/eval_sac/`. Architecture and compliance documentation: `docs/文件/ARCHITECTURE.md`, `docs/Semi-Shield_專案實驗與成果總說明.md`.

---

## Figure panel plan (4–6 figures, Nature-style)

| Fig | Proposed content | Existing asset |
|-----|------------------|----------------|
| 1 | Semi-Shield architecture: OT ingest → evidence chain → judgment → CORG release | `figure1_architecture.png` |
| 2 | Binding comparison: citation vs obligation read-set under mutation | `figure2_corg_release.png` |
| 3 | FAR vs resolution scatter or curve for B0–B5 | `risk_resolution_curve.png` |
| 4 | Paired FAR delta B5−B4 with bootstrap CI | *generate from results* |
| 5 | Release unauthorized rate by binding | *bar chart from Table 5* |
| 6 | Latency p50 by baseline | `latency_p50.png` |

---

## Author checklist before external submission

- [ ] Replace simulated κ with expert dual annotation  
- [ ] Run Ollama B2 writer subset; report end-to-end latency  
- [ ] Address irrelevant-still-release metric or reframe claims  
- [ ] Clarify graph retrieval negative result or improve topology model  
- [ ] Register dataset manifest DOI or supplemental zip (`data/eval_sac/`)  

---

*End of Nature-format draft*
