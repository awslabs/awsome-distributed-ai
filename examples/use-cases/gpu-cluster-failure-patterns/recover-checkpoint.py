#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from pathlib import Path

path = Path('/checkpoints')
if (path / '.aim344-fixture').read_text().strip() != 'AIM344 isolated checkpoint fixture':
    raise SystemExit('Refusing recovery outside the isolated lab fixture.')
for name in ('aim344-filler.bin', 'next.json'):
    (path / name).unlink(missing_ok=True)
print('Removed lab filler and partial write; retained last.json.')
