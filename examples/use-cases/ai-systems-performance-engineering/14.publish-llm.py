#!/usr/bin/env python3
"""Publish a completed measurement through the lab's existing Pushgateway helper."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from llm_metrics import observability_values
from metrics import push


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--pushgateway', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--instance-type', required=True)
    args = parser.parse_args()
    if not args.pushgateway or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', value) for value in (
            args.run_id, args.configuration, args.instance_type)):
        parser.error('nonempty Pushgateway URL and simple identifiers required')
    result = json.loads(args.result.read_text())
    values = observability_values(result)
    kind = ('campaign' if result.get('metric') == 'Training goodput' else
            'serving' if result.get('metrics', {}).get('metric') == 'Serving goodput' else 'training_segment')
    if any(value is None or not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError('nonfinite, missing or negative measurement')
    push(args.pushgateway, args.run_id, args.configuration, args.instance_type, values, measurement_kind=kind)
    print(json.dumps(values, indent=2))


if __name__ == '__main__':
    main()
