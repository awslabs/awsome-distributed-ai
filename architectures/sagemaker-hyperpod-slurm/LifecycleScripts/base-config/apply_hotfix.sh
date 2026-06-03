#!/bin/bash

echo 'BEGIN: apply hotfix'

# Future proof in case node type is needed.
[[ "$1" == "" ]] && NODE_TYPE=other || NODE_TYPE="$1"

set -exuo pipefail

BIN_DIR=$(dirname $(realpath ${BASH_SOURCE[@]}))

# By default, apply all available hotfix scripts.
declare -a PKGS_SCRIPTS=(
    $BIN_DIR/hotfix/*.sh
)
## To apply specific hotfix only, uncomment below stanza, then edit the scripts list.
# declare -a PKGS_SCRIPTS=(
#     $BIN_DIR/hotfix/hold-lustre-client.sh
#     $BIN_DIR/hotfix/mock-gpu-driver-deb.sh
# )

# Save a few 'apt-get update' on hotfix scripts (unless they add new repo).
apt-get -y -o DPkg::Lock::Timeout=120 update

for i in $BIN_DIR/hotfix/*.sh; do
    bash -x $i
done

echo 'END: apply hotfix'
