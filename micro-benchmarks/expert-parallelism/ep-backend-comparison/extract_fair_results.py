#!/usr/bin/env python3
"""Extract canonical JSONL result records from a mixed benchmark log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_result_io import PREFIX, load_result_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_log", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args()

    results = load_result_log(args.input_log)
    if not results:
        raise SystemExit(f"no fair results found in {args.input_log}")
    args.output_jsonl.write_text(
        "".join(PREFIX + json.dumps(result, sort_keys=True) + "\n" for result in results)
    )
    print(f"extracted {len(results)} fair results from {args.input_log}")


if __name__ == "__main__":
    main()
