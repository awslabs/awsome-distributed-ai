#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Check the complete correctness sweep without inventing a bandwidth threshold."""
import argparse
from pathlib import Path
import re


def summarize(text):
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 13 or not fields[0].isdigit() or fields[2] != 'float':
            continue
        rows.append(fields)
    if {int(row[0]) for row in rows} != {2 ** power for power in range(3, 32)}:
        raise ValueError('Incomplete sweep: require every doubling from 8 B through 2 GiB.')
    for row in rows:
        if row[8] != '0' or row[12] != '0':
            raise ValueError('Correctness failed or disabled: both #wrong columns must contain 0 mismatches.')
    if not re.search(r'Out of bounds values\s*:\s*0\s+OK', text):
        raise ValueError('Missing successful out-of-bounds summary.')
    final = next(row for row in rows if int(row[0]) == 2 ** 31)
    return (f'2 GiB: out-of-place busbw={float(final[7]):.2f} GB/s; '
            f'in-place busbw={float(final[11]):.2f} GB/s; 0 mismatches in both columns. '
            'Correctness PASS. Confirm transport from NCCL logs and EFA counter deltas separately.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    try:
        print(summarize(args.log.read_text()))
    except ValueError as error:
        raise SystemExit(str(error)) from error
