# SAC CORG 評估結果（本機執行）

- 執行時間：2.23s
- Packets：522（groups=120）

## Table 2 — Dataset composition

| 項目 | 值 |
|------|-----|
| actual_groups | 120 |
| actual_packets | 522 |
| SUPPORTED | 240 |
| VIOLATED | 42 |
| INSUFFICIENT | 240 |

## Table 4 — OT judgment (FAR ↓, resolution ↑)

| Method | FAR | Resolution | Macro-F1 |
|--------|-----|------------|----------|
| B0 | 0.5745 | 0.8511 | 0.4714 |
| B0-full | 0.0000 | 0.5035 | 0.6533 |
| B2 | 0.4326 | 0.6383 | 0.4436 |
| B3 | 0.4078 | 0.5213 | 0.4355 |
| B4 | 0.4078 | 0.5213 | 0.4355 |
| B5 | 0.0000 | 0.5035 | 0.6533 |

## Table 5 — Release (uncited counterevidence mutation)

| Binding | Unauthorized release rate | Irrelevant still releases |
|---------|---------------------------|---------------------------|
| AGT-cited | 1.0 | 1.0 |
| AGT-global | 0.0 | 1.0 |
| AGT+CORG | 0.0 | 1.0 |
| AGT+same_R | 0.0 | 1.0 |

## Table 6 — Key contrasts

- **B5 vs B4** (FAR): Δ=-0.407801
- **B3 vs B2** (resolution): Δ=-0.117021
- **AGT+CORG vs AGT-cited** (unauthorized_release_rate): Δ=1.0

完整 JSON：`C:\Users\Max\Desktop\ot_test\data\eval_sac\results.json`