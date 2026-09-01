#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Common bats setup/teardown for deploy.sh unit tests.
# Loaded by each .bats file via: load 'helpers/setup'

# Project root (parent of tests/)
PROJECT_DIR="$(cd "$(dirname "${BATS_TEST_FILENAME}")/.." && pwd)"
FIXTURE_DIR="${PROJECT_DIR}/tests/fixtures"
TEST_TEMP_DIR=""

setup() {
    # Guard: check that bats helper libraries are installed
    if [[ ! -d "${PROJECT_DIR}/tests/bats/bats-support" ]] || \
       [[ ! -d "${PROJECT_DIR}/tests/bats/bats-assert" ]]; then
        echo "Error: bats helper libraries not found." >&2
        echo "  Run: bash tests/install_bats_libs.sh" >&2
        return 1
    fi

    # Load bats helpers
    load 'bats/bats-support/load'
    load 'bats/bats-assert/load'

    # Create temp directory for each test
    TEST_TEMP_DIR="$(mktemp -d)"

    # Source the helpers library
    source "${PROJECT_DIR}/lib/deploy_helpers.sh"

    # Load and activate AWS mock
    source "${PROJECT_DIR}/tests/helpers/mock_aws.bash"
    mock_aws
}

teardown() {
    # Clean up temp directory
    if [[ -n "${TEST_TEMP_DIR}" && -d "${TEST_TEMP_DIR}" ]]; then
        rm -rf "${TEST_TEMP_DIR}"
    fi
}
