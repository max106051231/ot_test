# ISO27001 多輪對話測試報告

- 時間：2026-08-25T07:46:19.655558+00:00
- 模型數：15
- 失敗回合：45 / 105

## 問題序列

1. 我們現在符合 ISO27001 嗎？
2. 為什麼你不能確定？
3. 哪些 evidence 不足？
4. 哪一台設備造成 monitoring coverage 下降？
5. 給我圖表。
6. 這會影響哪些控制項？
7. 下一步要做什麼？

## Gemma 2B OT (`gemma_2b_ot`)

通過：**5/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 1.7 | 3 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.5 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 1.6 | 3 | — |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 0.9 | 3 | — |
| q5_chart | 給我圖表。 | blocked | 0.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 1.6 | 3 | — |
| q7_next_steps | 下一步要做什麼？ | success | 1.1 | 3 | — |

## Llama 3.2 3B OT (`llama32_3b_ot`)

通過：**5/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 2.2 | 3 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 3.3 | 3 | — |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 1.7 | 3 | — |
| q5_chart | 給我圖表。 | blocked | 0.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 2.5 | 3 | — |
| q7_next_steps | 下一步要做什麼？ | success | 1.3 | 3 | — |

## Phi-4 Mini OT (`phi4_mini_ot`)

通過：**5/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 7.1 | 3 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.8 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 8.1 | 3 | — |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 4.1 | 3 | — |
| q5_chart | 給我圖表。 | blocked | 0.8 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 7.3 | 3 | — |
| q7_next_steps | 下一步要做什麼？ | success | 4.8 | 3 | — |

## Qwen2.5-1.5B OT (`qwen25_1p5b_ot`)

通過：**5/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 1.6 | 3 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.2 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 2.2 | 3 | — |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 1.5 | 3 | — |
| q5_chart | 給我圖表。 | blocked | 0.2 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 1.7 | 3 | — |
| q7_next_steps | 下一步要做什麼？ | success | 1.5 | 3 | — |

## Qwen2.5-3B OT (`qwen25_3b_ot`)

通過：**5/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 1.7 | 3 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.2 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 1.7 | 3 | — |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 1.0 | 3 | — |
| q5_chart | 給我圖表。 | blocked | 0.2 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 1.7 | 3 | — |
| q7_next_steps | 下一步要做什麼？ | success | 1.3 | 3 | — |

## Qwen2.5-7B OT (`qwen25_7b_ot`)

通過：**5/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 3.2 | 3 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.4 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 2.7 | 3 | — |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 1.4 | 3 | — |
| q5_chart | 給我圖表。 | blocked | 0.4 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 2.1 | 3 | — |
| q7_next_steps | 下一步要做什麼？ | success | 1.3 | 3 | — |

## Qwen3-4B OT (`qwen3_4b_ot`)

通過：**3/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 8.7 | 3 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 2.9 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 6.1 | 3 | — |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 6.0 | 3 | — |
| q5_chart | 給我圖表。 | blocked | 2.6 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 6.5 | 3 | bad_fallback, off_topic |
| q7_next_steps | 下一步要做什麼？ | success | 6.5 | 3 | bad_fallback |

## Gemma 2B (`gemma2:2b`)

通過：**5/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 1.9 | 0 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 2.8 | 0 | — |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 1.3 | 0 | — |
| q5_chart | 給我圖表。 | blocked | 0.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 1.4 | 0 | — |
| q7_next_steps | 下一步要做什麼？ | success | 1.5 | 0 | — |

## Llama 3.2 3B (`llama3.2:3b`)

通過：**5/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 2.6 | 0 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 2.6 | 0 | — |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 3.2 | 0 | — |
| q5_chart | 給我圖表。 | blocked | 0.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 2.0 | 0 | — |
| q7_next_steps | 下一步要做什麼？ | success | 2.0 | 0 | — |

## Phi-4 Mini (`phi4`)

SKIP: timed out

## Qwen2.5-1.5B (`qwen2.5:1.5b`)

通過：**5/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 0.9 | 0 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 1.1 | 0 | — |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 1.0 | 0 | — |
| q5_chart | 給我圖表。 | blocked | 0.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 3.5 | 0 | — |
| q7_next_steps | 下一步要做什麼？ | success | 2.3 | 0 | — |

## Qwen2.5-3B (`qwen2.5:3b`)

通過：**4/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 1.8 | 0 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 1.2 | 0 | off_topic |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 1.5 | 0 | — |
| q5_chart | 給我圖表。 | blocked | 0.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 1.2 | 0 | — |
| q7_next_steps | 下一步要做什麼？ | success | 1.4 | 0 | — |

## Qwen2.5-7B (`qwen2.5:7b`)

通過：**4/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 2.0 | 0 | — |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.5 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 1.9 | 0 | off_topic |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 2.0 | 0 | — |
| q5_chart | 給我圖表。 | blocked | 0.4 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 1.5 | 0 | — |
| q7_next_steps | 下一步要做什麼？ | success | 1.4 | 0 | — |

## Qwen3-14B (`qwen3:14b`)

通過：**4/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 2.7 | 0 | off_topic |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 0.6 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 2.8 | 0 | — |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 2.1 | 0 | — |
| q5_chart | 給我圖表。 | blocked | 0.5 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 2.3 | 0 | — |
| q7_next_steps | 下一步要做什麼？ | success | 2.2 | 0 | — |

## Qwen3-4B (`qwen3:4b`)

通過：**0/7**

| # | 問題 | 結果 | 秒 | RAG | 問題 |
|---|------|------|-----|-----|------|
| q1_iso_compliant | 我們現在符合 ISO27001 嗎？ | success | 6.5 | 0 | bad_fallback, off_topic |
| q2_why_uncertain | 為什麼你不能確定？ | blocked | 3.3 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q3_evidence_gap | 哪些 evidence 不足？ | success | 7.0 | 0 | bad_fallback, off_topic |
| q4_coverage_device | 哪一台設備造成 monitoring cover | success | 7.4 | 0 | bad_fallback |
| q5_chart | 給我圖表。 | blocked | 3.5 | 0 | guardrail_blocked, bad_fallback, guardrail_blocked |
| q6_controls | 這會影響哪些控制項？ | success | 8.2 | 0 | bad_fallback, off_topic |
| q7_next_steps | 下一步要做什麼？ | success | 8.3 | 0 | bad_fallback |
