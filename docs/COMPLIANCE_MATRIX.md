# Semi-Shield ISO 27001 合規矩陣與技術實施

## 設計原則

**No Evidence, No Compliance Claim** — 每項 ISO 控制項宣稱必須具備 `evidence_id`、hash chain，且 LLM 診斷經 Human Review 後方可納入稽核報告。

## Annex A 覆蓋（Phase 1 MVP）

| 項目 | 數量 |
|------|------|
| Annex A 總計 | 93 |
| 矩陣已定義（A.5 + A.8） | 71 |
| Phase 1 in_scope | ~50 |
| OT syslog 自動監控 | 6 |

自動化對映：`access_control`→A.5.15、`supplier_security`→A.5.19、`malware_defense`→A.8.7、`patch_management`→A.8.8、`recipe_audit`→A.8.19、`sec_gem_log`→A.8.24

## 五層架構

Web Console → Guardrails + Human Review → Compliance Matrix → RAG & LLM → Agent Workflow

## Agent Workflow

Collector → Auditor → Reviewer → Reporter（`agent_orchestrator.py`，Phase 2：LangGraph/CrewAI）

## Guardrails

RoBERTa 微調 + 硬性規則 + allowlist + borderline Human Review + 輸出脫敏

## GRC 對標

Vanta / Drata / Secureframe / OneTrust / ServiceNow GRC / Hyperproof / Microsoft Compliance Manager — 詳見 `compliance/grc_positioning.json`

## KPI 目標

- Annex A 覆蓋 ≥80%
- 對映準確率 ≥85%
- Human review 採納率 ≥75%
- Evidence 追溯 100%
- 報告 ≤2 天（基線 4 週）

## 操作

- 架構頁：http://localhost:2000/compliance
- 管線：`POST /api/compliance/pipeline`
- 矩陣：`GET /api/compliance/matrix`
