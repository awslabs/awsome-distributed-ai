#!/usr/bin/env bats
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Unit tests for destroy.sh
# Run: bats tests/test_destroy.bats

load 'helpers/setup'

###########################
## destroy.sh arg parsing #
###########################

@test "destroy.sh: --help exits 0 and prints usage" {
    run bash "${PROJECT_DIR}/destroy.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

@test "destroy.sh: fails when --infra is missing" {
    run bash "${PROJECT_DIR}/destroy.sh"
    assert_failure
    assert_output --partial "Error: --infra is required"
}

@test "destroy.sh: fails with invalid --infra value" {
    run bash "${PROJECT_DIR}/destroy.sh" --infra docker
    assert_failure
    assert_output --partial "Error: --infra must be 'cfn' or 'tf'"
}

@test "destroy.sh: fails with unknown option" {
    run bash "${PROJECT_DIR}/destroy.sh" --foobar
    assert_failure
    assert_output --partial "Error: Unknown option"
}

@test "destroy.sh: --help mentions --region flag" {
    run bash "${PROJECT_DIR}/destroy.sh" --help
    assert_success
    assert_output --partial "--region"
}

@test "destroy.sh: --help mentions --stack-name flag" {
    run bash "${PROJECT_DIR}/destroy.sh" --help
    assert_success
    assert_output --partial "--stack-name"
}

@test "destroy.sh: aborts when confirmation prompt is non-interactive" {
    # In a non-interactive shell, read from /dev/null returns empty,
    # which doesn't match [Yy], so the script should abort.
    run bash -c "echo 'n' | bash '${PROJECT_DIR}/destroy.sh' --infra cfn"
    assert_success
    assert_output --partial "Aborted"
}

###########################
## teardown order #########
###########################

@test "destroy.sh: LB Controller uninstall comes after namespace deletion" {
    local ns_line lb_line
    ns_line=$(grep -n 'Delete Namespaces' "${PROJECT_DIR}/destroy.sh" | head -1 | cut -d: -f1)
    lb_line=$(grep -n 'Uninstall LB Controller' "${PROJECT_DIR}/destroy.sh" | head -1 | cut -d: -f1)
    [[ "${ns_line}" -lt "${lb_line}" ]]
}

@test "destroy.sh: cert-manager uninstall comes after LB Controller" {
    local lb_line cert_line
    lb_line=$(grep -n 'Uninstall LB Controller' "${PROJECT_DIR}/destroy.sh" | head -1 | cut -d: -f1)
    cert_line=$(grep -n 'Uninstall cert-manager' "${PROJECT_DIR}/destroy.sh" | head -1 | cut -d: -f1)
    [[ "${lb_line}" -lt "${cert_line}" ]]
}

@test "destroy.sh: cert-manager uninstall comes before CodeBuild stack" {
    local cert_line cb_line
    cert_line=$(grep -n 'Uninstall cert-manager' "${PROJECT_DIR}/destroy.sh" | head -1 | cut -d: -f1)
    cb_line=$(grep -n 'Delete CodeBuild Stack' "${PROJECT_DIR}/destroy.sh" | head -1 | cut -d: -f1)
    [[ "${cert_line}" -lt "${cb_line}" ]]
}

###########################
## local keyword fix ######
###########################

@test "destroy.sh: no local keyword outside functions" {
    # The 'local' keyword must only appear inside function bodies.
    # destroy.sh has no functions, so 'local' should not appear at all.
    run grep -n '^\s*local ' "${PROJECT_DIR}/destroy.sh"
    assert_failure
}

###########################
## IAM role/policy names ##
###########################

@test "destroy.sh: defines LB_CONTROLLER_IAM_ROLE_NAME" {
    run grep 'LB_CONTROLLER_IAM_ROLE_NAME=' "${PROJECT_DIR}/destroy.sh"
    assert_success
    assert_output --partial 'AmazonEKS_LB_Controller_Role_slinky'
}

@test "destroy.sh: defines LB_CONTROLLER_IAM_POLICY_NAME" {
    run grep 'LB_CONTROLLER_IAM_POLICY_NAME=' "${PROJECT_DIR}/destroy.sh"
    assert_success
    assert_output --partial 'AWSLoadBalancerControllerIAMPolicy_slinky'
}

@test "destroy.sh: defines EBS_CSI_IAM_ROLE_NAME" {
    run grep 'EBS_CSI_IAM_ROLE_NAME=' "${PROJECT_DIR}/destroy.sh"
    assert_success
    assert_output --partial 'AmazonEKS_EBS_CSI_DriverRole_slinky'
}

@test "destroy.sh: EBS CSI uninstall comes before LB Controller" {
    local ebs_line lb_line
    ebs_line=$(grep -n 'Uninstall EBS CSI' "${PROJECT_DIR}/destroy.sh" | head -1 | cut -d: -f1)
    lb_line=$(grep -n 'Uninstall LB Controller' "${PROJECT_DIR}/destroy.sh" | head -1 | cut -d: -f1)
    [[ "${ebs_line}" -lt "${lb_line}" ]]
}

@test "destroy.sh: deletes gp3 StorageClass" {
    run grep 'storageclass gp3' "${PROJECT_DIR}/destroy.sh"
    assert_success
}

###########################
## cleanup files ##########
###########################

@test "destroy.sh: cleans up lustre-pvc-slurm.yaml" {
    run grep 'lustre-pvc-slurm.yaml' "${PROJECT_DIR}/destroy.sh"
    assert_success
}

@test "destroy.sh: deletes FSx PVC during teardown" {
    run grep 'Delete FSx PVC' "${PROJECT_DIR}/destroy.sh"
    assert_success
}

###########################
## EKS_CLUSTER_NAME warns #
###########################

@test "destroy.sh: warns when EKS_CLUSTER_NAME unset for EBS CSI cleanup" {
    run grep -A2 'WARNING.*EKS_CLUSTER_NAME.*EBS CSI' "${PROJECT_DIR}/destroy.sh"
    assert_success
    assert_output --partial "manual deletion"
}

@test "destroy.sh: warns when EKS_CLUSTER_NAME unset for LB Controller cleanup" {
    run grep -A2 'WARNING.*EKS_CLUSTER_NAME.*LB Controller' "${PROJECT_DIR}/destroy.sh"
    assert_success
    assert_output --partial "manual deletion"
}
