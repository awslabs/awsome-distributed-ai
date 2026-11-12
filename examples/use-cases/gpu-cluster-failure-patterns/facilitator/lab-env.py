#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Print the site's lab.env from the cluster's values.

    python3 facilitator/lab-env.py PARTITION=gpu-g7 CPUS_PER_RANK=24 \\
        AIM344_EXPECTED_EFA_DEVICES=2 NCCL_SOCKET_IFNAME==enp71s0 \\
        AIM344_TARGET_NODE=gpu-g7-1 AIM344_PEER_NODE=gpu-g7-2 \\
        AIM344_EFA_DEVICE=rdmap83s0 \\
        AIM344_SLURM_BIN=/opt/aws/pcs/scheduler/slurm-25.11/bin > lab.env

Install the output as lab.env in the lab directory on the login node and on
both GPU nodes. Each line keeps a value that the caller's environment already
sets, so a participant can still override one for a run. PARTITION is required;
every other value has a default or is optional.
"""
import re
import shlex
import sys


class Refusal(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Refusal(message)


def lab_environment(overrides):
    """The text of lab.env: the defaults below, updated with `overrides`."""
    values = dict(LAB_DIR='/opt/aim344/device-recovery/companion',
                  GPUS_PER_NODE='8', CPUS_PER_RANK='24', MEMORY_PER_NODE='0',
                  NCCL_ENROOT_IMAGE='/opt/aim344/nccl-baseline.sqsh',
                  TORCH_IMAGE='/opt/aim344/aim344.sqsh',
                  CHECKPOINT_DIR='/run/aim344-checkpoints', FI_EFA_USE_HUGE_PAGE='0',
                  AIM344_SLURM_BIN='/opt/aws/pcs/scheduler/slurm-25.05/bin')
    # The seat's handout values and Slurm client directory, which the login
    # profile exports: AIM344_TARGET_NODE, AIM344_PEER_NODE, AIM344_EFA_DEVICE
    # (the target's efa_rdma_device) and AIM344_SLURM_BIN.
    allowed = {'AIM344_TARGET_NODE', 'AIM344_PEER_NODE', 'AIM344_EFA_DEVICE', 'AIM344_SLURM_BIN',
               'PARTITION', 'NCCL_CONTAINER', 'NCCL_TESTS_BIN', 'NCCL_MPI',
               'NCCL_TIMEOUT', 'NCCL_ISOLATION_TESTS', 'NCCL_ISOLATION_TIMEOUT',
               'AIM344_CHECK5_TIMEOUT', 'AIM344_CONTAINER_PREP_TIMEOUT',
               'NCCL_SOCKET_IFNAME', 'OMPI_MCA_pml', 'OMPI_MCA_btl',
               'OMPI_MCA_btl_tcp_if_exclude', 'PMIX_MCA_gds', 'OFI_NCCL_PROTOCOL', 'NCCL_PROTO',
               'FI_EFA_USE_HUGE_PAGE', 'FI_PROVIDER', 'NCCL_DEBUG', 'TORCH_IMAGE',
               'NCCL_ENROOT_IMAGE', 'AIM344_EXPECTED_EFA_DEVICES', 'CPUS_PER_RANK'}
    require(isinstance(overrides, dict) and set(overrides) <= allowed,
            'Unsupported lab.env value: '
            + ', '.join(sorted(set(overrides) - allowed if isinstance(overrides, dict) else [])))
    require(all(isinstance(v, str) and v and '\n' not in v for v in overrides.values()),
            'Invalid lab.env value')
    values.update(overrides)
    if 'AIM344_EXPECTED_EFA_DEVICES' in overrides:
        count = overrides['AIM344_EXPECTED_EFA_DEVICES']
        require(re.fullmatch(r'[1-9][0-9]?', count) and int(count) <= 32,
                'Invalid assigned EFA count')
        values['HEALTHCHECK_EXPECTED_EFA_COUNT'] = count
    # Match the native allocation parser's positive decimal contract. The site
    # supplies capacity; do not impose a new instance-specific CPU ceiling.
    require(re.fullmatch(r'[1-9][0-9]*', values['CPUS_PER_RANK']),
            'Invalid assigned CPUs per rank')
    for key in ('NCCL_CONTAINER', 'NCCL_ENROOT_IMAGE', 'TORCH_IMAGE'):
        if key in values:
            require(values[key] in ('/opt/aim344/aim344.sqsh', '/opt/aim344/nccl-baseline.sqsh'),
                    'Verification image must be a provisioned pinned object')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', values.get('PARTITION', '')),
            'Supply the assigned PARTITION')
    values['AIM344_SUITE_ENTRY'] = ('/opt/aim344-healthcheck/validation/gpu-cluster-healthcheck/'
                                    'gpu-healthcheck.sh')
    # Assignment defaults survive clean sudo/login environments when sourced by
    # the participant; existing explicit caller values retain precedence.
    return ''.join(f'export {key}=${{{key}:-{shlex.quote(value)}}}\n'
                   for key, value in values.items())


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    overrides = {}
    try:
        for argument in arguments:
            key, separator, value = argument.partition('=')
            require(separator and key not in overrides,
                    f'Give each value once as KEY=VALUE: {argument!r}')
            overrides[key] = value
        text = lab_environment(overrides)
    except Refusal as refusal:
        sys.stderr.write(f'{refusal}\n')
        return 2
    sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
