#!/usr/bin/env bats
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Unit tests for setup.sh
# Run: bats tests/test_setup.bats

load 'helpers/setup'

###########################
## setup.sh arg parsing ###
###########################

@test "setup.sh: --help exits 0 and prints usage" {
    run bash "${PROJECT_DIR}/setup.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

@test "setup.sh: fails when --instance-type is missing" {
    run bash "${PROJECT_DIR}/setup.sh" --infra cfn
    assert_failure
    assert_output --partial "Error: --instance-type is required"
}

@test "setup.sh: fails when --infra is missing" {
    run bash "${PROJECT_DIR}/setup.sh" --instance-type ml.g5.8xlarge
    assert_failure
    assert_output --partial "Error: --infra is required"
}

@test "setup.sh: fails with invalid --infra value" {
    run bash "${PROJECT_DIR}/setup.sh" --instance-type ml.g5.8xlarge --infra docker
    assert_failure
    assert_output --partial "Error: --infra must be 'cfn' or 'tf'"
}

@test "setup.sh: fails with unknown option" {
    run bash "${PROJECT_DIR}/setup.sh" --foobar
    assert_failure
    assert_output --partial "Error: Unknown option"
}

@test "setup.sh: --skip-build flag is accepted in usage" {
    run bash "${PROJECT_DIR}/setup.sh" --help
    assert_success
    assert_output --partial "--skip-build"
}

@test "setup.sh: --local-build flag is accepted in usage" {
    run bash "${PROJECT_DIR}/setup.sh" --help
    assert_success
    assert_output --partial "--local-build"
}

@test "setup.sh: --help mentions --instance-count flag" {
    run bash "${PROJECT_DIR}/setup.sh" --help
    assert_success
    assert_output --partial "--instance-count"
}

###########################
## resolve_helm_profile ###
## (integration tests)  ###
###########################

@test "resolve_helm_profile: ml.g5.8xlarge sets all 7 template variables" {
    resolve_helm_profile "ml.g5.8xlarge" 4
    assert_equal "${HELM_ACCEL_INSTANCE_TYPE}" "ml.g5.8xlarge"
    assert_equal "${GPU_COUNT}" "1"
    assert_equal "${EFA_COUNT}" "1"
    assert_equal "${GPU_GRES}" "gpu:a10g:1"
    assert_equal "${REPLICAS}" "4"
    assert_equal "${MGMT_INSTANCE_TYPE}" "ml.m5.4xlarge"
    assert_equal "${PVC_NAME}" "fsx-claim"
}

@test "resolve_helm_profile: ml.p5.48xlarge sets correct overrides" {
    resolve_helm_profile "ml.p5.48xlarge" 2
    assert_equal "${HELM_ACCEL_INSTANCE_TYPE}" "ml.p5.48xlarge"
    assert_equal "${GPU_COUNT}" "8"
    assert_equal "${EFA_COUNT}" "32"
    assert_equal "${GPU_GRES}" "gpu:h100:8"
    assert_equal "${REPLICAS}" "2"
}

@test "resolve_helm_profile: invalid instance type returns 1" {
    run resolve_helm_profile "ml.g5.99xlarge"
    assert_failure
    assert_output --partial "is not a valid instance type"
}

###########################
## values template ########
## substitution tests   ###
###########################

@test "values template: g5 substitution produces no unresolved variables" {
    resolve_helm_profile "ml.g5.8xlarge" 4

    sed -e "s|\${image_repository}|123456789012.dkr.ecr.us-west-2.amazonaws.com/dlc-slurmd|g" \
        -e "s|\${image_tag}|25.11.1-ubuntu24.04|g" \
        -e "s|\${ssh_key}|ssh-ed25519 AAAAC3test test@example.com|g" \
        -e "s|\${mgmt_instance_type}|${MGMT_INSTANCE_TYPE}|g" \
        -e "s|\${accel_instance_type}|${HELM_ACCEL_INSTANCE_TYPE}|g" \
        -e "s|\${gpu_count}|${GPU_COUNT}|g" \
        -e "s|\${efa_count}|${EFA_COUNT}|g" \
        -e "s|\${gpu_gres}|${GPU_GRES}|g" \
        -e "s|\${replicas}|${REPLICAS}|g" \
        -e "s|\${pvc_name}|${PVC_NAME}|g" \
        "${FIXTURE_DIR}/slurm-values.yaml.template" > "${TEST_TEMP_DIR}/slurm-values.yaml"

    # No unsubstituted template variables remain
    run grep -c '${' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_failure
}

@test "values template: p5 substitution has correct GPU and EFA counts" {
    resolve_helm_profile "ml.p5.48xlarge" 2

    sed -e "s|\${image_repository}|123456789012.dkr.ecr.us-west-2.amazonaws.com/dlc-slurmd|g" \
        -e "s|\${image_tag}|25.11.1-ubuntu24.04|g" \
        -e "s|\${ssh_key}|ssh-ed25519 AAAAC3test test@example.com|g" \
        -e "s|\${mgmt_instance_type}|${MGMT_INSTANCE_TYPE}|g" \
        -e "s|\${accel_instance_type}|${HELM_ACCEL_INSTANCE_TYPE}|g" \
        -e "s|\${gpu_count}|${GPU_COUNT}|g" \
        -e "s|\${efa_count}|${EFA_COUNT}|g" \
        -e "s|\${gpu_gres}|${GPU_GRES}|g" \
        -e "s|\${replicas}|${REPLICAS}|g" \
        -e "s|\${pvc_name}|${PVC_NAME}|g" \
        "${FIXTURE_DIR}/slurm-values.yaml.template" > "${TEST_TEMP_DIR}/slurm-values.yaml"

    run grep 'gpu:h100:8' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'vpc.amazonaws.com/efa: 32' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'replicas: 2' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success
}
