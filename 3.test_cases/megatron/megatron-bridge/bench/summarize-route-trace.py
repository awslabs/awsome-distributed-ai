#!/usr/bin/env python3
"""Hash and histogram the discarded first-iteration routing trace."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys


def main() -> None:
    source = pathlib.Path(sys.argv[1])
    destination = pathlib.Path(sys.argv[2])
    digest = hashlib.sha256()
    histogram: dict[int, int] = {}
    records = 0
    selections = 0
    for path in sorted(source.glob("router_trace_rank*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            record = json.loads(line)
            indices = record.get("top_indices")
            if indices is None:
                continue
            canonical = json.dumps(indices, separators=(",", ":")).encode()
            digest.update(canonical)
            records += 1
            for row in indices:
                for expert in row:
                    histogram[int(expert)] = histogram.get(int(expert), 0) + 1
                    selections += 1
    summary = {
        "schema_version": 1,
        "source": str(source),
        "records": records,
        "selections": selections,
        "topk_index_sha256": digest.hexdigest() if records else None,
        "tokens_per_expert": {str(key): histogram[key] for key in sorted(histogram)},
        "max_to_mean_load": max(histogram.values()) / (selections / len(histogram)) if histogram else None,
    }
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not records:
        raise SystemExit("no routing trace records")
    print("ROUTE_SUMMARY " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
