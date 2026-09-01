#!/usr/bin/env bats
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Unit tests for deploy.sh and lib/deploy_helpers.sh
# Run: bats tests/test_deploy.bats

load 'helpers/setup'

###########################
## resolve_instance_profile
###########################

@test "resolve_instance_profile: ml.g5.8xlarge sets correct type and default count" {
    resolve_instance_profile "ml.g5.8xlarge"
    assert_equal "${ACCEL_INSTANCE_TYPE}" "ml.g5.8xlarge"
    assert_equal "${ACCEL_INSTANCE_COUNT}" "4"
}

@test "resolve_instance_profile: ml.p5.48xlarge with count 2" {
    resolve_instance_profile "ml.p5.48xlarge" 2
    assert_equal "${ACCEL_INSTANCE_TYPE}" "ml.p5.48xlarge"
    assert_equal "${ACCEL_INSTANCE_COUNT}" "2"
}

@test "resolve_instance_profile: custom count overrides default" {
    resolve_instance_profile "ml.g5.8xlarge" 8
    assert_equal "${ACCEL_INSTANCE_COUNT}" "8"
}

@test "resolve_instance_profile: empty instance type returns 1" {
    run resolve_instance_profile ""
    assert_failure
    assert_output --partial "Error: --instance-type is required"
}

@test "resolve_instance_profile: missing ml. prefix returns 1" {
    run resolve_instance_profile "g5.8xlarge"
    assert_failure
    assert_output --partial "must start with 'ml.'"
}

@test "resolve_instance_profile: invalid instance type returns 1" {
    run resolve_instance_profile "ml.g5.99xlarge"
    assert_failure
    assert_output --partial "is not a valid instance type"
}

###########################
## resolve_helm_profile ###
###########################

@test "resolve_helm_profile: ml.g5.8xlarge sets correct GPU/EFA/GRES/replicas" {
    resolve_helm_profile "ml.g5.8xlarge" 4
    assert_equal "${HELM_ACCEL_INSTANCE_TYPE}" "ml.g5.8xlarge"
    assert_equal "${GPU_COUNT}" "1"
    assert_equal "${EFA_COUNT}" "1"
    assert_equal "${GPU_GRES}" "gpu:a10g:1"
    assert_equal "${REPLICAS}" "4"
    assert_equal "${MGMT_INSTANCE_TYPE}" "ml.m5.4xlarge"
    assert_equal "${PVC_NAME}" "fsx-claim"
}

@test "resolve_helm_profile: ml.p5.48xlarge sets correct GPU/EFA/GRES/replicas" {
    resolve_helm_profile "ml.p5.48xlarge" 2
    assert_equal "${HELM_ACCEL_INSTANCE_TYPE}" "ml.p5.48xlarge"
    assert_equal "${GPU_COUNT}" "8"
    assert_equal "${EFA_COUNT}" "32"
    assert_equal "${GPU_GRES}" "gpu:h100:8"
    assert_equal "${REPLICAS}" "2"
}

@test "resolve_helm_profile: ml.g6.12xlarge auto-discovers 4 L4 GPUs" {
    resolve_helm_profile "ml.g6.12xlarge" 3
    assert_equal "${GPU_COUNT}" "4"
    assert_equal "${GPU_GRES}" "gpu:l4:4"
    assert_equal "${EFA_COUNT}" "1"
    assert_equal "${REPLICAS}" "3"
}

@test "resolve_helm_profile: replicas defaults to 4 when count not specified" {
    resolve_helm_profile "ml.g5.8xlarge"
    assert_equal "${REPLICAS}" "4"
}

@test "resolve_helm_profile: rejects Neuron/Trainium instances" {
    run resolve_helm_profile "ml.trn1.32xlarge" 4
    assert_failure
    assert_output --partial "Neuron/Trainium instances are not currently supported"
}

@test "resolve_helm_profile: rejects CPU-only instances" {
    run resolve_helm_profile "ml.m5.xlarge" 2
    assert_failure
    assert_output --partial "has no GPUs"
}

@test "resolve_helm_profile: rejects invalid instance type" {
    run resolve_helm_profile "ml.g5.99xlarge" 4
    assert_failure
    assert_output --partial "is not a valid instance type"
}

@test "resolve_helm_profile: empty instance type returns 1" {
    run resolve_helm_profile ""
    assert_failure
    assert_output --partial "Error: --instance-type is required"
}

@test "resolve_helm_profile: missing ml. prefix returns 1" {
    run resolve_helm_profile "p5.48xlarge"
    assert_failure
    assert_output --partial "must start with 'ml.'"
}

###########################
## resolve_training_plan ##
###########################

@test "resolve_training_plan: valid plan sets ARN and AZ ID" {
    run resolve_training_plan "test-plan" "us-west-2"
    assert_success
    assert_output --partial "arn:aws:sagemaker:us-west-2:123456789012:training-plan/test-plan"
    assert_output --partial "Active"
    # Verify exported variables are set (run executes in subshell, so check via function call)
    resolve_training_plan "test-plan" "us-west-2"
    assert_equal "${TRAINING_PLAN_ARN}" "arn:aws:sagemaker:us-west-2:123456789012:training-plan/test-plan"
    assert_equal "${TRAINING_PLAN_AZ_ID}" "usw2-az2"
}

@test "resolve_training_plan: nonexistent plan returns 1" {
    run resolve_training_plan "nonexistent-plan" "us-west-2"
    assert_failure
    assert_output --partial "not found"
}

@test "resolve_training_plan: empty name returns 1" {
    run resolve_training_plan "" "us-west-2"
    assert_failure
    assert_output --partial "requires a plan name"
}

@test "resolve_training_plan: failed plan returns 1" {
    run resolve_training_plan "failed-plan" "us-west-2"
    assert_failure
    assert_output --partial "status 'Failed'"
}

@test "resolve_training_plan: expired plan warns but succeeds" {
    run resolve_training_plan "expired-plan" "us-west-2"
    assert_success
    assert_output --partial "WARNING"
    assert_output --partial "Expired"
    # Verify ARN is still set
    resolve_training_plan "expired-plan" "us-west-2"
    assert_equal "${TRAINING_PLAN_ARN}" "arn:aws:sagemaker:us-west-2:123456789012:training-plan/expired-plan"
}

@test "resolve_training_plan: no reserved capacity returns 1" {
    run resolve_training_plan "no-capacity-plan" "us-west-2"
    assert_failure
    assert_output --partial "no reserved capacity"
}

###########################
## check_command ##########
###########################

@test "check_command: finds existing command (bash)" {
    run check_command "bash"
    assert_success
}

@test "check_command: fails for nonexistent command" {
    run check_command "definitely_not_a_real_command_xyz"
    assert_failure
    assert_output --partial "Error: 'definitely_not_a_real_command_xyz' is required but not installed."
}

@test "check_command: finds jq" {
    run check_command "jq"
    assert_success
}

###########################
## validate_az_id #########
###########################

@test "validate_az_id: returns 0 when AZ exists in list" {
    run validate_az_id "usw2-az2" "usw2-az1,usw2-az2,usw2-az3,usw2-az4"
    assert_success
}

@test "validate_az_id: returns 1 when AZ not in list" {
    run validate_az_id "usw2-az9" "usw2-az1,usw2-az2,usw2-az3,usw2-az4"
    assert_failure
}

@test "validate_az_id: handles single-element list" {
    run validate_az_id "usw2-az1" "usw2-az1"
    assert_success
}

@test "validate_az_id: no partial matches" {
    run validate_az_id "usw2-az1" "usw2-az10,usw2-az11"
    assert_failure
}

@test "validate_az_id: handles different regions" {
    run validate_az_id "use1-az2" "use1-az1,use1-az2,use1-az3"
    assert_success
}

###########################
## resolve_cfn_params #####
###########################

@test "resolve_cfn_params: substitutes AvailabilityZoneIds" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "use1-az1,use1-az2,use1-az3" \
        "use1-az2" \
        "ml.g5.8xlarge" \
        4)

    local az_ids
    az_ids=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "AvailabilityZoneIds") | .ParameterValue')

    assert_equal "${az_ids}" "use1-az1,use1-az2,use1-az3"
}

@test "resolve_cfn_params: substitutes FsxAvailabilityZoneId" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "use1-az1,use1-az2,use1-az3" \
        "use1-az2" \
        "ml.g5.8xlarge" \
        4)

    local fsx_az
    fsx_az=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "FsxAvailabilityZoneId") | .ParameterValue')

    assert_equal "${fsx_az}" "use1-az2"
}

@test "resolve_cfn_params: sets TargetAvailabilityZoneId in all instance groups" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "use1-az1,use1-az2,use1-az3" \
        "use1-az2" \
        "ml.g5.8xlarge" \
        4)

    local ig_json
    ig_json=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "InstanceGroupSettings1") | .ParameterValue')

    # Both instance groups should have use1-az2
    local az_values
    az_values=$(echo "${ig_json}" | jq -r '.[].TargetAvailabilityZoneId')

    # All values should be use1-az2
    while IFS= read -r line; do
        assert_equal "${line}" "use1-az2"
    done <<< "${az_values}"
}

@test "resolve_cfn_params: default keeps ml.g5.8xlarge in accelerated group" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.g5.8xlarge" \
        4)

    local ig_json
    ig_json=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "InstanceGroupSettings1") | .ParameterValue')

    local accel_type
    accel_type=$(echo "${ig_json}" | jq -r \
        '.[] | select(.InstanceGroupName == "accelerated-instance-group-1") | .InstanceType')

    assert_equal "${accel_type}" "ml.g5.8xlarge"
}

@test "resolve_cfn_params: overrides accelerated group to ml.p5.48xlarge" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.p5.48xlarge" \
        2)

    local ig_json
    ig_json=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "InstanceGroupSettings1") | .ParameterValue')

    local accel_type
    accel_type=$(echo "${ig_json}" | jq -r \
        '.[] | select(.InstanceGroupName == "accelerated-instance-group-1") | .InstanceType')

    assert_equal "${accel_type}" "ml.p5.48xlarge"
}

@test "resolve_cfn_params: overrides accelerated group instance count" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.p5.48xlarge" \
        2)

    local ig_json
    ig_json=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "InstanceGroupSettings1") | .ParameterValue')

    local accel_count
    accel_count=$(echo "${ig_json}" | jq -r \
        '.[] | select(.InstanceGroupName == "accelerated-instance-group-1") | .InstanceCount')

    assert_equal "${accel_count}" "2"
}

@test "resolve_cfn_params: general group unchanged after accelerated override" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.p5.48xlarge" \
        2)

    local ig_json
    ig_json=$(echo "${result}" | jq -r \
        '.[] | select(.ParameterKey == "InstanceGroupSettings1") | .ParameterValue')

    local general_type
    general_type=$(echo "${ig_json}" | jq -r \
        '.[] | select(.InstanceGroupName == "general-instance-group-2") | .InstanceType')

    local general_count
    general_count=$(echo "${ig_json}" | jq -r \
        '.[] | select(.InstanceGroupName == "general-instance-group-2") | .InstanceCount')

    assert_equal "${general_type}" "ml.m5.4xlarge"
    assert_equal "${general_count}" "2"
}

@test "resolve_cfn_params: fails when params file not found" {
    run resolve_cfn_params \
        "/nonexistent/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.g5.8xlarge" \
        4

    assert_failure
    assert_output --partial "Error: Parameters file not found"
}

@test "resolve_cfn_params: preserves all 40 parameters" {
    local result
    result=$(resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" \
        "usw2-az2" \
        "ml.g5.8xlarge" \
        4)

    local count
    count=$(echo "${result}" | jq 'length')

    assert_equal "${count}" "40"
}

@test "resolve_cfn_params: injects TrainingPlanArn into accelerated group" {
    local result
    result=$(resolve_cfn_params "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" "usw2-az2" "ml.p5.48xlarge" 2 \
        "arn:aws:sagemaker:us-west-2:123456789012:training-plan/test-plan")

    # Extract the accelerated group from InstanceGroupSettings1
    local accel_arn
    accel_arn=$(echo "${result}" | jq -r '
        .[] | select(.ParameterKey == "InstanceGroupSettings1") |
        .ParameterValue | fromjson |
        .[] | select(.InstanceGroupName == "accelerated-instance-group-1") |
        .TrainingPlanArn')

    assert_equal "${accel_arn}" "arn:aws:sagemaker:us-west-2:123456789012:training-plan/test-plan"
}

@test "resolve_cfn_params: omits TrainingPlanArn when empty" {
    local result
    result=$(resolve_cfn_params "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" "usw2-az2" "ml.g5.8xlarge" 4 "")

    # TrainingPlanArn should not be present
    local has_arn
    has_arn=$(echo "${result}" | jq -r '
        .[] | select(.ParameterKey == "InstanceGroupSettings1") |
        .ParameterValue | fromjson |
        .[] | select(.InstanceGroupName == "accelerated-instance-group-1") |
        has("TrainingPlanArn")')

    assert_equal "${has_arn}" "false"
}

@test "resolve_cfn_params: TrainingPlanArn absent from general group" {
    local result
    result=$(resolve_cfn_params "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" "usw2-az2" "ml.p5.48xlarge" 2 \
        "arn:aws:sagemaker:us-west-2:123456789012:training-plan/test-plan")

    # General group should NOT have TrainingPlanArn
    local has_arn
    has_arn=$(echo "${result}" | jq -r '
        .[] | select(.ParameterKey == "InstanceGroupSettings1") |
        .ParameterValue | fromjson |
        .[] | select(.InstanceGroupName == "general-instance-group-2") |
        has("TrainingPlanArn")')

    assert_equal "${has_arn}" "false"
}

###########################
## validate_cfn_template ##
###########################

@test "validate_cfn_template: succeeds when all params match template" {
    # Resolve params to a temp file (same as deploy_cfn does)
    local resolved_file="${TEST_TEMP_DIR}/resolved-params.json"
    resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" "usw2-az2" "ml.g5.8xlarge" 4 \
        > "${resolved_file}"

    run validate_cfn_template \
        "https://aws-sagemaker-hyperpod-cluster-setup-us-west-2-prod.s3.us-west-2.amazonaws.com/templates/main-stack-eks-based-template.yaml" \
        "${resolved_file}" \
        "us-west-2"

    assert_success
    assert_output --partial "Template syntax: OK"
    assert_output --partial "Validation: OK"
}

@test "validate_cfn_template: reports parameter counts" {
    local resolved_file="${TEST_TEMP_DIR}/resolved-params.json"
    resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" "usw2-az2" "ml.g5.8xlarge" 4 \
        > "${resolved_file}"

    run validate_cfn_template \
        "https://aws-sagemaker-hyperpod-cluster-setup-us-west-2-prod.s3.us-west-2.amazonaws.com/templates/main-stack-eks-based-template.yaml" \
        "${resolved_file}" \
        "us-west-2"

    assert_success
    assert_output --partial "40 provided, 40 in template"
}

@test "validate_cfn_template: fails when template URL is unreachable" {
    local resolved_file="${TEST_TEMP_DIR}/resolved-params.json"
    resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" "usw2-az2" "ml.g5.8xlarge" 4 \
        > "${resolved_file}"

    run validate_cfn_template \
        "https://invalid-bucket.s3.us-west-2.amazonaws.com/templates/missing.yaml" \
        "${resolved_file}" \
        "us-west-2"

    assert_failure
    assert_output --partial "Template validation failed"
}

@test "validate_cfn_template: fails when resolved params file not found" {
    run validate_cfn_template \
        "https://example.com/template.yaml" \
        "/nonexistent/params.json" \
        "us-west-2"

    assert_failure
    assert_output --partial "Resolved params file not found"
}

@test "validate_cfn_template: fails when template_url is empty" {
    run validate_cfn_template "" "/tmp/dummy.json" "us-west-2"

    assert_failure
    assert_output --partial "template_url is required"
}

@test "validate_cfn_template: warns about extra params not in template" {
    # Create params with an extra key the template doesn't know about
    local resolved_file="${TEST_TEMP_DIR}/resolved-params.json"
    resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" "usw2-az2" "ml.g5.8xlarge" 4 \
        > "${resolved_file}"

    # Append an extra parameter
    local modified_file="${TEST_TEMP_DIR}/modified-params.json"
    jq '. + [{"ParameterKey": "BogusParameter", "ParameterValue": "test"}]' \
        "${resolved_file}" > "${modified_file}"

    run validate_cfn_template \
        "https://aws-sagemaker-hyperpod-cluster-setup-us-west-2-prod.s3.us-west-2.amazonaws.com/templates/main-stack-eks-based-template.yaml" \
        "${modified_file}" \
        "us-west-2"

    # Should succeed but with a warning
    assert_success
    assert_output --partial "WARNING: Parameters in params file not found in template"
    assert_output --partial "BogusParameter"
    assert_output --partial "Validation: OK"
}

@test "validate_cfn_template: fails when required params are missing" {
    # Create params file missing required keys (HyperPodClusterName has no default)
    local resolved_file="${TEST_TEMP_DIR}/resolved-params.json"
    resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" "usw2-az2" "ml.g5.8xlarge" 4 \
        > "${resolved_file}"

    # Remove HyperPodClusterName and ResourceNamePrefix (both required — no default)
    local stripped_file="${TEST_TEMP_DIR}/stripped-params.json"
    jq 'map(select(.ParameterKey != "HyperPodClusterName" and .ParameterKey != "ResourceNamePrefix"))' \
        "${resolved_file}" > "${stripped_file}"

    run validate_cfn_template \
        "https://aws-sagemaker-hyperpod-cluster-setup-us-west-2-prod.s3.us-west-2.amazonaws.com/templates/main-stack-eks-based-template.yaml" \
        "${stripped_file}" \
        "us-west-2"

    assert_failure
    assert_output --partial "Required template parameters missing"
    assert_output --partial "HyperPodClusterName"
    assert_output --partial "ResourceNamePrefix"
}

@test "validate_cfn_template: succeeds when optional params are omitted" {
    # Remove params that have defaults — should still pass
    local resolved_file="${TEST_TEMP_DIR}/resolved-params.json"
    resolve_cfn_params \
        "${FIXTURE_DIR}/params.json" \
        "usw2-az1,usw2-az2" "usw2-az2" "ml.g5.8xlarge" 4 \
        > "${resolved_file}"

    # Remove StorageCapacity (has default "1200")
    local reduced_file="${TEST_TEMP_DIR}/reduced-params.json"
    jq 'map(select(.ParameterKey != "StorageCapacity"))' \
        "${resolved_file}" > "${reduced_file}"

    run validate_cfn_template \
        "https://aws-sagemaker-hyperpod-cluster-setup-us-west-2-prod.s3.us-west-2.amazonaws.com/templates/main-stack-eks-based-template.yaml" \
        "${reduced_file}" \
        "us-west-2"

    assert_success
    assert_output --partial "Validation: OK"
}

###########################
## resolve_tf_vars ########
###########################

@test "resolve_tf_vars: overrides aws_region" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-east-1" "use1-az2" "ml.g5.8xlarge" 4

    run grep 'aws_region' "${target}"
    assert_output --partial 'us-east-1'
}

@test "resolve_tf_vars: overrides availability_zone_id" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-east-1" "use1-az2" "ml.g5.8xlarge" 4

    run grep 'availability_zone_id' "${target}"
    # Both instance groups should have use1-az2
    assert_output --partial 'use1-az2'
    # Verify no leftover usw2-az2
    run grep 'usw2-az2' "${target}"
    assert_failure
}

@test "resolve_tf_vars: default instance type preserved when unchanged" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-west-2" "usw2-az2" "ml.g5.8xlarge" 4

    run grep 'ml.g5.8xlarge' "${target}"
    assert_success
}

@test "resolve_tf_vars: overrides accelerated instance type to ml.p5.48xlarge" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-west-2" "usw2-az2" "ml.p5.48xlarge" 2

    run grep 'ml.p5.48xlarge' "${target}"
    assert_success
}

@test "resolve_tf_vars: overrides accelerated instance count" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-west-2" "usw2-az2" "ml.p5.48xlarge" 2

    # The first instance_count should be 2 (accelerated group).
    local first_count
    first_count=$(awk '/instance_count/ { match($0, /[0-9]+/); print substr($0, RSTART, RLENGTH); exit }' "${target}")

    assert_equal "${first_count}" "2"
}

@test "resolve_tf_vars: does not change general group instance type" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-west-2" "usw2-az2" "ml.p5.48xlarge" 2

    run grep 'ml.m5.4xlarge' "${target}"
    assert_success
}

@test "resolve_tf_vars: overrides to arbitrary instance type" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-west-2" "usw2-az2" "ml.g6.12xlarge" 6

    run grep 'ml.g6.12xlarge' "${target}"
    assert_success

    local first_count
    first_count=$(awk '/instance_count/ { match($0, /[0-9]+/); print substr($0, RSTART, RLENGTH); exit }' "${target}")
    assert_equal "${first_count}" "6"
}

@test "resolve_tf_vars: fails when file not found" {
    run resolve_tf_vars "/nonexistent/custom.tfvars" "us-west-2" "usw2-az2" "ml.g5.8xlarge" 4
    assert_failure
    assert_output --partial "Error: tfvars file not found"
}

@test "resolve_tf_vars: cleans up sed .bak files" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-east-1" "use1-az2" "ml.g5.8xlarge" 4

    # No .bak files should remain
    run ls "${TEST_TEMP_DIR}"/*.bak 2>/dev/null
    assert_failure
}

@test "resolve_tf_vars: injects training_plan_arn into first group" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-west-2" "usw2-az2" "ml.p5.48xlarge" 2 \
        "arn:aws:sagemaker:us-west-2:123456789012:training-plan/test-plan"

    # training_plan_arn should appear in the file
    run grep 'training_plan_arn' "${target}"
    assert_success
    assert_output --partial "arn:aws:sagemaker:us-west-2:123456789012:training-plan/test-plan"
}

@test "resolve_tf_vars: omits training_plan_arn when empty" {
    local target="${TEST_TEMP_DIR}/custom.tfvars"
    cp "${FIXTURE_DIR}/custom.tfvars" "${target}"

    resolve_tf_vars "${target}" "us-west-2" "usw2-az2" "ml.g5.8xlarge" 4 ""

    # training_plan_arn should NOT appear
    run grep 'training_plan_arn' "${target}"
    assert_failure
}

###########################
## deploy.sh arg parsing ##
###########################

@test "deploy.sh: --help exits 0 and prints usage" {
    run bash "${PROJECT_DIR}/deploy.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

@test "deploy.sh: fails when --instance-type is missing" {
    run bash "${PROJECT_DIR}/deploy.sh" --infra cfn
    assert_failure
    assert_output --partial "Error: --instance-type is required"
}

@test "deploy.sh: fails when --infra is missing" {
    run bash "${PROJECT_DIR}/deploy.sh" --instance-type ml.g5.8xlarge
    assert_failure
    assert_output --partial "Error: --infra is required"
}

@test "deploy.sh: fails with invalid --infra value" {
    run bash "${PROJECT_DIR}/deploy.sh" --instance-type ml.g5.8xlarge --infra docker
    assert_failure
    assert_output --partial "Error: --infra must be 'cfn' or 'tf'"
}

@test "deploy.sh: fails with unknown option" {
    run bash "${PROJECT_DIR}/deploy.sh" --foobar
    assert_failure
    assert_output --partial "Error: Unknown option"
}

@test "deploy.sh: --help mentions --instance-type flag" {
    run bash "${PROJECT_DIR}/deploy.sh" --help
    assert_output --partial "--instance-type"
}

@test "deploy.sh: --help mentions --instance-count flag" {
    run bash "${PROJECT_DIR}/deploy.sh" --help
    assert_output --partial "--instance-count"
}

@test "deploy.sh: --help mentions --training-plan flag" {
    run bash "${PROJECT_DIR}/deploy.sh" --help
    assert_output --partial "--training-plan"
}

###########################
## values template sed ####
###########################

@test "values template: g5 substitution produces valid YAML" {
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

    # Verify key values were substituted
    run grep 'ml.g5.8xlarge' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'ml.m5.4xlarge' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'gpu:a10g:1' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'replicas: 4' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    # No unsubstituted template variables remain
    run grep -c '${' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_failure
}

@test "values template: p5 substitution has correct instance type and GPU count" {
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

    run grep 'ml.p5.48xlarge' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'gpu:h100:8' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'replicas: 2' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success

    run grep 'vpc.amazonaws.com/efa: 32' "${TEST_TEMP_DIR}/slurm-values.yaml"
    assert_success
}

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

###########################
## install.sh arg parsing #
###########################

@test "install.sh: --help exits 0 and prints usage" {
    run bash "${PROJECT_DIR}/install.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

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

###########################
## CFN idempotency ########
###########################

@test "deploy.sh: deploy_cfn checks for existing stack before create" {
    run grep 'DOES_NOT_EXIST' "${PROJECT_DIR}/deploy.sh"
    assert_success
}

@test "deploy.sh: deploy_cfn uses update-stack for existing stacks" {
    run grep 'update-stack' "${PROJECT_DIR}/deploy.sh"
    assert_success
}

@test "deploy.sh: deploy_cfn handles 'No updates' gracefully" {
    run grep 'No updates are to be performed' "${PROJECT_DIR}/deploy.sh"
    assert_success
}

@test "deploy.sh: deploy_cfn rejects stacks in bad states" {
    run grep 'Cannot create or update' "${PROJECT_DIR}/deploy.sh"
    assert_success
}
