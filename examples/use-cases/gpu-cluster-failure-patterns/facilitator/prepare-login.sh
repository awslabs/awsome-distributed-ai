#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$LAB_DIR/pins.env"
stage=${AIM344_STAGE_DIR:-/fsx/aim344}
# Node-local staging is an explicit facilitator choice. Copy the prepared tree
# and images to the same absolute path on every assigned compute node.
if [[ ${AIM344_NODE_LOCAL:-0} == 1 ]]; then
    [[ $stage == /opt/* && -d $stage ]] || { echo 'Node-local staging requires an existing /opt directory' >&2; exit 2; }
else
    findmnt -T "$stage" -n -o FSTYPE | grep -qx lustre
fi
mkdir -p "$stage/evidence"
command -v docker enroot git sha256sum
# Login imports run as the participant user. Keep their paths independent of
# root's pre-existing Enroot directories; compute-node Pyxis keeps its defaults.
private_dir=/tmp/aim344-enroot-$(id -u)
install -d -m 0700 "$private_dir" "$private_dir/cache" "$private_dir/data" "$private_dir/runtime"
export ENROOT_CACHE_PATH="$private_dir/cache" ENROOT_DATA_PATH="$private_dir/data" ENROOT_RUNTIME_PATH="$private_dir/runtime"
export ENROOT_MAX_PROCESSORS=${ENROOT_MAX_PROCESSORS:-8}
cd "$LAB_DIR"
for name in aim344 nccl-baseline; do
    image="$stage/$name.sqsh"
    if [[ ! -s $image ]]; then
        [[ -w $stage ]] || { echo "Stage directory is not writable: $stage" >&2; exit 1; }
        if [[ $name == aim344 ]]; then source_image=${AIM344_TORCH_SOURCE:-}; else source_image=${AIM344_BASELINE_SOURCE:-}; fi
        if [[ -n $source_image ]]; then
            [[ $source_image == *@sha256:* ]] || { echo 'Prebuilt images require an immutable digest' >&2; exit 1; }
            docker pull "$source_image"
            docker tag "$source_image" "aim344:$name"
        else
            docker build --target "$name" --build-arg NCCL_BUILD_JOBS="$ENROOT_MAX_PROCESSORS" -t "aim344:$name" .
        fi
        enroot import -o "$image" "dockerd://aim344:$name"
    fi
    sha256sum "$image"
done
# Fetch the immutable health-suite commit when the companion is a shallow clone.
repo=$(git rev-parse --show-toplevel)
if ! git -C "$repo" cat-file -e "$HEALTHCHECK_COMMIT^{commit}" 2>/dev/null; then
    git -C "$repo" fetch --depth=1 origin "$HEALTHCHECK_COMMIT"
fi
if [[ ! -s $stage/healthcheck-pinned.tgz ]]; then
    git -C "$repo" archive "$HEALTHCHECK_COMMIT" validation/gpu-cluster-healthcheck | gzip > "$stage/healthcheck-pinned.tgz"
fi
# Compare file contents and executable modes; extraction ownership is host-local.
expected=$(mktemp /tmp/aim344-health-source.XXXXXX.tar)
git -C "$repo" archive "$HEALTHCHECK_COMMIT" validation/gpu-cluster-healthcheck > "$expected"
python3 - "$expected" "$stage/healthcheck-pinned.tgz" <<'PY'
import hashlib, sys, tarfile
def entries(path):
    with tarfile.open(path) as archive:
        return {member.name: (member.mode, hashlib.sha256(archive.extractfile(member).read()).hexdigest())
                for member in archive if member.isfile()}
assert entries(sys.argv[1]) == entries(sys.argv[2]), 'Health archive differs from the pinned tree'
print('Health archive contents and file modes match the pinned tree')
PY
if [[ ! -d $stage/healthcheck-source ]]; then
    mkdir "$stage/healthcheck-source"
    tar -xzf "$stage/healthcheck-pinned.tgz" -C "$stage/healthcheck-source"
fi
sha256sum "$stage/healthcheck-pinned.tgz"
printf 'Prepared images and pinned health suite in %s\n' "$stage"
