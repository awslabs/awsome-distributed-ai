#!/usr/bin/env python3
"""Compare continuous measurements with explicit same-work invariants."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from llm_metrics import paired_training_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--serving-experiment', choices=['placement', 'server_batch', 'client_load'],
                      help='compare saved serving measurements with fixed workload and resource budget')
    mode.add_argument('--numerics-policy', type=Path, help='compare correctness-run directories using these preregistered limits')

    args = parser.parse_args()
    if args.serving_experiment:
        from serving_goodput import paired_serving_report
        result = paired_serving_report(json.loads(args.before.read_text()),
                                       json.loads(args.after.read_text()), experiment=args.serving_experiment)
        print(json.dumps(result, indent=2))
        return
    if args.numerics_policy:
        import torch
        from llm_numerics import numerical_report
        torch.set_num_threads(4)
        result = numerical_report(args.before, args.after, json.loads(args.numerics_policy.read_text()))
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['passed'] else 1)
    result = paired_training_report(json.loads(args.before.read_text()),
                                    json.loads(args.after.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
