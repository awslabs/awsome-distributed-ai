#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
# Run only in the lab's private container, once per node, with no GPU job active.
plugin=/opt/amazon/ofi-nccl/lib
saved=/opt/amazon/ofi-nccl/lib.aim344-disabled
case ${1:-} in
    hide)
        [[ -f $plugin/libnccl-net-ofi.so && ! -e $saved ]] || { echo 'Unexpected plugin state.' >&2; exit 1; }
        mv -- "$plugin" "$saved"
        ;;
    restore)
        if [[ -e $saved ]]; then
            [[ ! -e $plugin ]] || { echo 'Refusing to overwrite an existing plugin.' >&2; exit 1; }
            mv -- "$saved" "$plugin"
        fi
        [[ -f $plugin/libnccl-net-ofi.so ]]
        ;;
    require-present) [[ -f $plugin/libnccl-net-ofi.so && ! -e $saved ]] ;;
    *) echo 'Usage: plugin-state.sh hide|restore|require-present' >&2; exit 2 ;;
esac
printf 'host=%s plugin_action=%s\n' "$(hostname)" "$1"
