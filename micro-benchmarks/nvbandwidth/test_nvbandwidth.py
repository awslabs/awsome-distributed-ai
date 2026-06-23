# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Parse nvbandwidth output and gate on a per-direction bandwidth floor.

Usage:
    NVBW_MIN_GBPS=600 python3 test_nvbandwidth.py results.txt
or in CI, pipe nvbandwidth stdout into it:
    nvbandwidth -t host_to_device_memcpy_ce | NVBW_MIN_GBPS=600 python3 test_nvbandwidth.py -

The floor is a CONFIGURABLE threshold, not a fabricated GB200 spec number. Calibrate
NVBW_MIN_GBPS to your measured baseline (the observed C2C read point is ~821 GB/s; set
the floor below your own first-run number with margin).
"""
import os
import re
import sys

MIN_GBPS = float(os.environ.get("NVBW_MIN_GBPS", "600"))

# nvbandwidth prints a matrix of GB/s values; capture every float that looks like a
# bandwidth cell (skip the index header row/column).
_CELL = re.compile(r"\b(\d+\.\d+)\b")


def parse(text: str):
    values = []
    for line in text.splitlines():
        # data rows start with an integer device index then float cells
        if re.match(r"^\s*\d+\s+\d+\.\d+", line):
            values.extend(float(x) for x in _CELL.findall(line))
    return values


def main(argv):
    if len(argv) != 2:
        print("usage: test_nvbandwidth.py <results.txt|->", file=sys.stderr)
        return 2
    text = sys.stdin.read() if argv[1] == "-" else open(argv[1]).read()

    # IMEX preflight: the multi-node tests require the cross-instance domain to be UP.
    if "Domain State: UP" not in text and "multinode" in text:
        print("FAIL: IMEX domain not reported UP for a multinode run", file=sys.stderr)
        return 1

    cells = parse(text)
    if not cells:
        print("FAIL: no bandwidth values parsed (did nvbandwidth run?)", file=sys.stderr)
        return 1

    worst = min(cells)
    print(f"parsed {len(cells)} bandwidth cells; min={worst:.1f} GB/s; floor={MIN_GBPS:.1f} GB/s")
    if worst < MIN_GBPS:
        print(f"FAIL: a link is below the floor ({worst:.1f} < {MIN_GBPS:.1f} GB/s) -- "
              f"check for a miscabled NVSwitch port pair, a C2C link in warning state, "
              f"or a partial IMEX domain", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
