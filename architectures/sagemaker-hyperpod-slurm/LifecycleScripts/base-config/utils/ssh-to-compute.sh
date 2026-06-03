#!/bin/bash

set -exuo pipefail

[[ -f /opt/slurm/etc/slurm.conf ]] \
    && SLURM_CONFIG=/opt/slurm/etc/slurm.conf \
    || SLURM_CONFIG=/var/spool/slurmd/conf-cache/slurm.conf

# https://github.com/aws-samples/aws-efa-nccl-baseami-pipeline/blob/9d8a9273f72d7dee36f7f3e5e8a968b5e0f5f21b/nvidia-efa-ami_base/nvidia-efa-ml-ubuntu2004.yml#L163-L169
cat << EOF >> /etc/ssh/ssh_config.d/initsmhp-ssh.conf
Host 127.0.0.1 localhost $(hostname)
    StrictHostKeyChecking no
    HostbasedAuthentication no
    CheckHostIP no
    UserKnownHostsFile /dev/null

Match host * exec "grep '^NodeName=%h ' $SLURM_CONFIG &> /dev/null"
    StrictHostKeyChecking no
    HostbasedAuthentication no
    CheckHostIP no
    UserKnownHostsFile /dev/null
EOF
