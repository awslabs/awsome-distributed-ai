#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Install bats helper libraries (bats-assert, bats-support) into tests/bats/.
# These are not committed to the repo — run this once after cloning.
#
# Usage:
#   bash tests/install_bats_libs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATS_DIR="${SCRIPT_DIR}/bats"

mkdir -p "${BATS_DIR}"

install_lib() {
    local name="$1"
    local repo="$2"
    local target="${BATS_DIR}/${name}"

    if [[ -d "${target}" ]]; then
        echo "  ${name}: already installed, skipping"
        return 0
    fi

    echo "  ${name}: cloning from ${repo}..."
    git clone --depth 1 "${repo}" "${target}" 2>&1 | sed 's/^/    /'
    rm -rf "${target}/.git"
    echo "  ${name}: done"
}

echo "Installing bats helper libraries into ${BATS_DIR}/..."
echo ""
install_lib "bats-support" "https://github.com/bats-core/bats-support"
install_lib "bats-assert" "https://github.com/bats-core/bats-assert"
echo ""
echo "Done. You can now run: bats tests/test_deploy.bats"
