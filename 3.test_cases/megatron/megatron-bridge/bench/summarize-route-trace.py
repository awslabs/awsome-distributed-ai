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
    selection_slots = 0
    valid_selections = 0
    dropped_selections = 0
    expected_selections = 0
    for path in sorted(source.glob("router_trace_rank*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            record = json.loads(line)
            indices = record.get("top_indices")
            if indices is None:
                continue
            canonical = json.dumps(indices, separators=(",", ":")).encode()
            digest.update(canonical)
            records += 1
            num_tokens = int(record.get("num_tokens", len(indices)))
            topk = record.get("topk")
            if topk is None:
                topk = len(indices[0]) if indices else 0
            expected_selections += num_tokens * int(topk)
            for row in indices:
                for expert in row:
                    selection_slots += 1
                    expert = int(expert)
                    if expert < 0:
                        dropped_selections += 1
                        continue
                    histogram[expert] = histogram.get(expert, 0) + 1
                    valid_selections += 1
    no_token_drop = (
        records > 0
        and dropped_selections == 0
        and selection_slots == expected_selections
        and valid_selections == expected_selections
    )
    summary = {
        "schema_version": 2,
        "source": str(source),
        "records": records,
        "selections": selection_slots,
        "expected_selections": expected_selections,
        "valid_selections": valid_selections,
        "dropped_selections": dropped_selections,
        "no_token_drop": no_token_drop,
        "topk_index_sha256": digest.hexdigest() if records else None,
        "tokens_per_expert": {str(key): histogram[key] for key in sorted(histogram)},
        "max_to_mean_load": (
            max(histogram.values()) / (valid_selections / len(histogram)) if histogram else None
        ),
    }
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not records:
        raise SystemExit("no routing trace records")
    print("ROUTE_SUMMARY " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
