# Semi-Shield: Obligation-Closed Evidence and Approval-Bound Release for OT Cybersecurity Assessments

SAC '27 measured tables (same captions as protocol draft).

Table 2: Dataset composition (measured held-out run).
Partition Target groups Actual groups / packets / S,V,I / agreement
Development 30 / / /
Held-out OT 120 120 / 522 / 240,42,240 / rule-gold
Release mutations paired with held-out trials; relevant; irrelevant 36+36 trials / see Table 5 / —
Transfer family to specify / / / (not run)

Table 4: OT judgment results (%). FAR lower is better; resolution, F1 and evidence-set recall higher are better.
Method FAR Resolution Macro-F1 Evidence recall p95 s
B0 57.45 85.11 47.14 — —
B0-full 0.00 50.35 65.33 — —
B2 43.26 63.83 44.36 — —
B3 40.78 52.13 43.55 — —
B4 40.78 52.13 43.55 — —
B5 0.00 50.35 65.33 — —

Table 5: Actual report-release results (measured rates).
Approval binding Unauthorized release Unchanged valid release Irrelevant-update release Re-review
AGT-cited 100.00 — 100.00 —
AGT-global 0.00 — 100.00 —
AGT+CORG 0.00 — 100.00 —
AGT+same_R 0.00 — 100.00 —

Table 6: Interpretation guide (observed effects; paired bootstrap 2000 draws).
Contrast Supportive pattern If absent Observed Δ [CI]
B5 vs B4 FAR lower with CORG at comparable resolution — -0.407801 [-0.4545, -0.3601]
B3 vs B2 graph retrieval improves answerable resolution — -0.117021 [-0.1625, -0.0742]
AGT+CORG vs AGT-cited positive delta means CORG blocks more unsafe releases — 1.0 —
