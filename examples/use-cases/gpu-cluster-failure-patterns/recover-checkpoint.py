#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Remove the filler and the partial write that Incident 2 leaves in /checkpoints."""
from pathlib import Path

# workload.py FILLER_NAME and the unfinished checkpoint.
for name in ('ckpt-shard-00017.partial', 'next.json'):
    (Path('/checkpoints') / name).unlink(missing_ok=True)
print('Removed the filler and the partial write; kept last.json.')
