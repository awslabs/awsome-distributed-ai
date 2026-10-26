#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Render the launch template's NetworkInterfaces list.

An EFA-enabled GPU instance carries one network interface per network card, and the number of
cards is a property of the instance type: 1 on g7e.12xlarge, 32 on p5.48xlarge. CloudFormation
cannot loop, so the list is written out to the largest supported card count and the entries a
type does not have resolve to AWS::NoValue.

Writing 31 near-identical blocks by hand is not reviewable in a diff, so this script emits them
and tests/lint-templates.sh fails when the committed block differs from the render. Edit this
script, run it, and paste the output between the markers in
assets/eks-add-gpu-nodegroup.yaml.

Two properties differ per instance type and both come from the NicLayout mapping in that
template rather than from a branch here:

  PrimaryEfa           whether network card 0 supports EFA (false on p6-b300, where card 0 is
                       ENA-only) -- read by the PrimarySupportsEfa condition
  SecondaryDeviceIndex the device index used on cards 1..N-1: 1 on p5, p6-b200, p4d and g7e,
                       0 on p6-b300. Device indices are per network card, so both are valid
                       requests; this follows the layout that ran on real B300 hardware in
                       architectures/aws-pcs/assets/add-cng-p6-b300.yaml

The card count thresholds below mirror the Cards*Plus conditions in the template. Adding an
instance type whose card count is not already a threshold means adding a condition there and a
threshold here.
"""

import sys

# Largest card count in the NicLayout mapping (p5.48xlarge).
MAX_CARDS = 32

# The condition that gates each card, by the smallest card count that includes it. Cards2Plus
# covers card 1, Cards4Plus covers cards 2-3, and so on: the condition is true when the
# instance type has at least that many network cards.
THRESHOLDS = [
    (2, "Cards2Plus"),
    (4, "Cards4Plus"),
    (8, "Cards8Plus"),
    (16, "Cards16Plus"),
    (17, "Cards17Plus"),
    (32, "Cards32"),
]

INDENT = " " * 10
GROUPS = "Groups: [!Ref NodeSecurityGroupId, !Ref ClusterSecurityGroupId]"


def condition_for(card: int) -> str:
    for threshold, name in THRESHOLDS:
        if card < threshold:
            return name
    raise ValueError(f"no condition covers network card {card}")


def render() -> str:
    lines = [
        f"{INDENT}# Card 0 carries EFA on every supported type except p6-b300, where it is ENA-only.",
        f"{INDENT}- DeviceIndex: 0",
        f"{INDENT}  NetworkCardIndex: 0",
        f"{INDENT}  InterfaceType: !If [PrimarySupportsEfa, efa, !Ref 'AWS::NoValue']",
        f"{INDENT}  {GROUPS}",
    ]
    for card in range(1, MAX_CARDS):
        lines += [
            f"{INDENT}- !If",
            f"{INDENT}  - {condition_for(card)}",
            f"{INDENT}  - DeviceIndex: !If [SecondaryDeviceIndexIsZero, 0, 1]",
            f"{INDENT}    NetworkCardIndex: {card}",
            f"{INDENT}    InterfaceType: efa",
            f"{INDENT}    {GROUPS}",
            f"{INDENT}  - !Ref 'AWS::NoValue'",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.stdout.write(render())
