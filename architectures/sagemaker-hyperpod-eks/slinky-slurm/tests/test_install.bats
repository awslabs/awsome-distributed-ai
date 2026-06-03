#!/usr/bin/env bats
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Unit tests for install.sh
# Run: bats tests/test_install.bats

load 'helpers/setup'

###########################
## install.sh arg parsing #
###########################

@test "install.sh: --help exits 0 and prints usage" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

@test "install.sh: --help mentions --skip-setup" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--skip-setup"
}

@test "install.sh: --help mentions pass-through options" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--instance-type"
    assert_output --partial "--instance-count"
    assert_output --partial "--infra"
    assert_output --partial "--local-build"
    assert_output --partial "--skip-build"
}

@test "install.sh: fails with unknown option" {
    run bash "${PROJECT_DIR}/install.sh" --foobar
    assert_failure
    assert_output --partial "Error: Unknown option"
}

@test "install.sh: --skip-setup fails when slurm-values.yaml is missing" {
    # Ensure slurm-values.yaml does not exist in the project directory.
    # The mock aws and mock kubectl won't be present, but the script should
    # fail before reaching any kubectl/helm calls because the values file
    # doesn't exist.
    local temp_project="${TEST_TEMP_DIR}/project"
    mkdir -p "${temp_project}"
    cp "${PROJECT_DIR}/install.sh" "${temp_project}/"

    run bash "${temp_project}/install.sh" --skip-setup
    assert_failure
    assert_output --partial "slurm-values.yaml not found"
}

###########################
## pass-through flags #####
###########################

@test "install.sh: accepts --repo-name as pass-through flag" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--repo-name"
}

@test "install.sh: accepts --tag as pass-through flag" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--tag"
}

@test "install.sh: accepts --region flag" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--region"
}

###########################
## version constants ######
###########################

@test "install.sh: defines CERT_MANAGER_VERSION" {
    run grep 'CERT_MANAGER_VERSION=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial '1.19.2'
}

@test "install.sh: defines LB_CONTROLLER_CHART_VERSION" {
    run grep 'LB_CONTROLLER_CHART_VERSION=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial '1.11.0'
}

@test "install.sh: defines LB_CONTROLLER_IAM_ROLE_NAME" {
    run grep 'LB_CONTROLLER_IAM_ROLE_NAME=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'AmazonEKS_LB_Controller_Role_slinky'
}

@test "install.sh: defines LB_CONTROLLER_IAM_POLICY_NAME" {
    run grep 'LB_CONTROLLER_IAM_POLICY_NAME=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'AWSLoadBalancerControllerIAMPolicy_slinky'
}

@test "install.sh: defines EBS_CSI_IAM_ROLE_NAME" {
    run grep 'EBS_CSI_IAM_ROLE_NAME=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'AmazonEKS_EBS_CSI_DriverRole_slinky'
}

@test "install.sh: defines EBS_CSI_INLINE_POLICY_NAME" {
    run grep 'EBS_CSI_INLINE_POLICY_NAME=' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'SageMakerHyperPodVolumeAccess'
}

###########################
## install order ##########
###########################

@test "install.sh: cert-manager is installed before LB Controller" {
    # Verify the cert-manager section appears before the LB Controller section
    local cert_line lb_line
    cert_line=$(grep -n 'Install cert-manager' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    lb_line=$(grep -n 'Install LB Controller' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    [[ "${cert_line}" -lt "${lb_line}" ]]
}

@test "install.sh: LB Controller is installed before EBS CSI Driver" {
    local lb_line ebs_line
    lb_line=$(grep -n 'Install LB Controller' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    ebs_line=$(grep -n 'Install EBS CSI Driver' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    [[ "${lb_line}" -lt "${ebs_line}" ]]
}

@test "install.sh: EBS CSI Driver is installed before FSx PVC" {
    local ebs_line fsx_line
    ebs_line=$(grep -n 'Install EBS CSI Driver' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    fsx_line=$(grep -n 'Apply FSx PVC' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    [[ "${ebs_line}" -lt "${fsx_line}" ]]
}

@test "install.sh: LB Controller is installed before MariaDB" {
    local lb_line mariadb_line
    lb_line=$(grep -n 'Install LB Controller' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    mariadb_line=$(grep -n 'Install MariaDB' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    [[ "${lb_line}" -lt "${mariadb_line}" ]]
}

@test "install.sh: FSx PVC is applied before MariaDB" {
    local fsx_line mariadb_line
    fsx_line=$(grep -n 'Apply FSx PVC' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    mariadb_line=$(grep -n 'Install MariaDB' "${PROJECT_DIR}/install.sh" | head -1 | cut -d: -f1)
    [[ "${fsx_line}" -lt "${mariadb_line}" ]]
}

@test "install.sh: requires EKS_CLUSTER_NAME from env_vars.sh" {
    run grep 'EKS_CLUSTER_NAME' "${PROJECT_DIR}/install.sh"
    assert_success
}

@test "install.sh: requires VPC_ID from env_vars.sh" {
    run grep 'VPC_ID' "${PROJECT_DIR}/install.sh"
    assert_success
}

###########################
## skip flags #############
###########################

@test "install.sh: --help mentions --skip-cert-manager" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--skip-cert-manager"
}

@test "install.sh: --help mentions --skip-lb-controller" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--skip-lb-controller"
}

@test "install.sh: --help mentions --skip-ebs-csi" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--skip-ebs-csi"
}

@test "install.sh: defines SKIP_CERT_MANAGER default" {
    run grep 'SKIP_CERT_MANAGER=false' "${PROJECT_DIR}/install.sh"
    assert_success
}

@test "install.sh: defines SKIP_LB_CONTROLLER default" {
    run grep 'SKIP_LB_CONTROLLER=false' "${PROJECT_DIR}/install.sh"
    assert_success
}

@test "install.sh: defines SKIP_EBS_CSI default" {
    run grep 'SKIP_EBS_CSI=false' "${PROJECT_DIR}/install.sh"
    assert_success
}

@test "install.sh: --skip-cert-manager sets SKIP_CERT_MANAGER=true" {
    run grep -A1 '\-\-skip-cert-manager)' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'SKIP_CERT_MANAGER=true'
}

@test "install.sh: --skip-lb-controller sets SKIP_LB_CONTROLLER=true" {
    run grep -A1 '\-\-skip-lb-controller)' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'SKIP_LB_CONTROLLER=true'
}

@test "install.sh: --skip-ebs-csi sets SKIP_EBS_CSI=true" {
    run grep -A1 '\-\-skip-ebs-csi)' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'SKIP_EBS_CSI=true'
}

@test "install.sh: cert-manager section has skip guard" {
    run grep -A1 'SKIP_CERT_MANAGER.*true' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'Skipping cert-manager'
}

@test "install.sh: LB Controller section has skip guard" {
    run grep -A1 'SKIP_LB_CONTROLLER.*true' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'Skipping subnet tagging and LB Controller'
}

@test "install.sh: EBS CSI section has skip guard" {
    run grep 'Skipping EBS CSI driver' "${PROJECT_DIR}/install.sh"
    assert_success
}

###########################
## existing cluster flags #
###########################

@test "install.sh: --help mentions --cluster-name" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--cluster-name"
}

@test "install.sh: --help mentions --vpc-id" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "--vpc-id"
}

@test "install.sh: --help shows bring your own cluster example" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "Bring your own cluster"
}

@test "install.sh: defines CLUSTER_NAME_OVERRIDE default" {
    run grep 'CLUSTER_NAME_OVERRIDE=""' "${PROJECT_DIR}/install.sh"
    assert_success
}

@test "install.sh: defines VPC_ID_OVERRIDE default" {
    run grep 'VPC_ID_OVERRIDE=""' "${PROJECT_DIR}/install.sh"
    assert_success
}

@test "install.sh: --cluster-name overrides EKS_CLUSTER_NAME" {
    run grep 'CLUSTER_NAME_OVERRIDE' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'EKS_CLUSTER_NAME'
}

@test "install.sh: --vpc-id overrides VPC_ID" {
    run grep 'VPC_ID_OVERRIDE' "${PROJECT_DIR}/install.sh"
    assert_success
    assert_output --partial 'VPC_ID'
}

@test "install.sh: error message mentions --cluster-name" {
    run grep '\-\-cluster-name' "${PROJECT_DIR}/install.sh"
    assert_success
    # Should appear in both the arg parser and the error hint
    assert_output --partial 'pass --cluster-name'
}

@test "install.sh: error message mentions --vpc-id" {
    run grep 'pass --vpc-id' "${PROJECT_DIR}/install.sh"
    assert_success
}

###########################
## helm upgrade --install #
###########################

@test "install.sh: uses helm upgrade --install for cert-manager" {
    run grep 'helm upgrade --install cert-manager' "${PROJECT_DIR}/install.sh"
    assert_success
}

@test "install.sh: uses helm upgrade --install for LB Controller" {
    run grep 'helm upgrade --install aws-load-balancer-controller' "${PROJECT_DIR}/install.sh"
    assert_success
}

@test "install.sh: uses helm upgrade --install for MariaDB operator" {
    run grep 'helm upgrade --install mariadb-operator' "${PROJECT_DIR}/install.sh"
    assert_success
}

@test "install.sh: uses helm upgrade --install for Slurm operator" {
    run grep 'helm upgrade --install slurm-operator' "${PROJECT_DIR}/install.sh"
    assert_success
}

@test "install.sh: Slurm cluster chart still uses helm status guard" {
    # The Slurm cluster itself is guarded to prevent accidental upgrades
    run grep 'helm status slurm -n slurm' "${PROJECT_DIR}/install.sh"
    assert_success
}

###########################
## IAM idempotency ########
###########################

@test "install.sh: LB Controller role re-attaches policy when already exists" {
    # When the IAM role already exists, the script should still try
    # attaching the policy (handles partial failure on previous run)
    local in_else=false
    while IFS= read -r line; do
        if [[ "${line}" =~ "IAM role already exists: \${LB_CONTROLLER_IAM_ROLE_NAME}" ]]; then
            in_else=true
        fi
        if [[ "${in_else}" == "true" && "${line}" =~ "attach-role-policy" ]]; then
            return 0
        fi
        # Stop looking after next major section
        if [[ "${in_else}" == "true" && "${line}" =~ "LB_ROLE_ARN=" ]]; then
            break
        fi
    done < "${PROJECT_DIR}/install.sh"
    fail "Expected attach-role-policy in IAM role 'already exists' branch"
}

@test "install.sh: EBS CSI role re-attaches policy when already exists" {
    local in_else=false
    while IFS= read -r line; do
        if [[ "${line}" =~ "IAM role already exists: \${EBS_CSI_IAM_ROLE_NAME}" ]]; then
            in_else=true
        fi
        if [[ "${in_else}" == "true" && "${line}" =~ "attach-role-policy" ]]; then
            return 0
        fi
        if [[ "${in_else}" == "true" && "${line}" =~ "EBS_CSI_ROLE_ARN=" ]]; then
            break
        fi
    done < "${PROJECT_DIR}/install.sh"
    fail "Expected attach-role-policy in EBS CSI IAM role 'already exists' branch"
}
