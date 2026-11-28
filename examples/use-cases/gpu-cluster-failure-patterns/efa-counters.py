#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Record real byte counters only. Missing counters are not zero traffic."""
import argparse
import json
from pathlib import Path
import socket
import time


def read_counters():
    values = {}
    for device in Path('/sys/class/infiniband').glob('*'):
        driver = device / 'device/driver'
        if not driver.exists() or driver.resolve().name != 'efa':
            continue
        for counter in device.glob('ports/*/hw_counters/*bytes*'):
            values[str(counter)] = int(counter.read_text().strip())
    if not values:
        raise SystemExit('EFA byte counters unavailable; traffic attribution is UNVALIDATED.')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['snapshot', 'delta'])
    parser.add_argument('prefix')
    args = parser.parse_args()
    host = socket.gethostname()
    instance_id = Path('/sys/devices/virtual/dmi/id/board_asset_tag').read_text().strip()
    instance_type = Path('/sys/devices/virtual/dmi/id/product_name').read_text().strip()
    # Some machine images preserve a duplicate hostname. EC2 instance IDs keep
    # snapshots distinct when Slurm mounts a shared results directory.
    if not instance_id.startswith('i-') or not instance_id[2:].isalnum():
        raise SystemExit('Cannot identify the EC2 instance from DMI.')
    path = Path(f'{args.prefix}.{instance_id}.json')
    current = read_counters()
    if args.action == 'snapshot':
        path.write_text(json.dumps({'monotonic_s': time.monotonic(), 'counters_bytes': current}, indent=2) + '\n')
        return
    before = json.loads(path.read_text())
    if current.keys() != before['counters_bytes'].keys():
        raise SystemExit('Counter set changed; repeat the measurement.')
    deltas = {key: value - before['counters_bytes'][key] for key, value in current.items()}
    if any(value < 0 for value in deltas.values()):
        raise SystemExit('Counter reset or wrap detected; repeat the measurement.')
    print(json.dumps({'host': host, 'instance_id': instance_id, 'instance_type': instance_type,
                      'interval_s': time.monotonic() - before['monotonic_s'],
                      'deltas_bytes': deltas}, sort_keys=True))


if __name__ == '__main__':
    main()
