#!/usr/bin/env python3
"""Quick check: JSONL import counts under ot/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from code.server.app import (  # noqa: E402
    _iter_jsonl_objects,
    _parse_jsonl_log_file,
    scan_ot_directory,
    ot_logs_dir,
)

def main() -> int:
    ot = ot_logs_dir()
    jsonl_files = sorted(ot.rglob("*.jsonl"))
    print(f"OT folder: {ot}")
    for path in jsonl_files:
        if path.name in {"evidence_registry.jsonl", "review_queue.jsonl", "rag_pairs.jsonl"}:
            continue
        recs = list(_iter_jsonl_objects(path))
        events, meta = _parse_jsonl_log_file(path)
        print(
            f"- {path.name}: json_records={meta.get('json_records')} "
            f"event_lines={meta.get('event_lines')} "
            f"skipped_no_message={meta.get('skipped_no_message')} "
            f"iter={len(recs)} sample={len(events)}"
        )

    data = scan_ot_directory()
    if data.get("error"):
        print("scan error:", data["error"])
        return 1
    stats = data.get("import_stats") or {}
    print(
        f"scan total_event_lines={stats.get('total_event_lines')} "
        f"parsed_logs_sample={len(data.get('parsed_logs') or [])}"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
