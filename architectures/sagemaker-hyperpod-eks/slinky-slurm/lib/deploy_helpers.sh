#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Extracted helper functions for deploy.sh
# This file is sourced by deploy.sh and can be sourced independently
# in tests for unit testing.

###########################
## Instance Profile Res ###
###########################

# resolve_instance_profile <instance_type> [instance_count]
# Validates the SageMaker instance type via the EC2 API and sets
# ACCEL_INSTANCE_TYPE and ACCEL_INSTANCE_COUNT.
# The instance_type must start with "ml." (SageMaker convention).
# Returns 0 on success, 1 on invalid/missing type.
resolve_instance_profile() {
    local instance_type="$1"
    local instance_count="${2:-4}"

    if [[ -z "${instance_type}" ]]; then
        echo "Error: --instance-type is required"
        return 1
    fi

    # Validate ml. prefix
    if [[ "${instance_type}" != ml.* ]]; then
        echo "Error: Instance type must start with 'ml.' (got: ${instance_type})"
        return 1
    fi

    # Strip ml. prefix for EC2 API lookup
    local ec2_type="${instance_type#ml.}"

    if ! aws ec2 describe-instance-types \
        --instance-types "${ec2_type}" &>/dev/null; then
        echo "Error: '${instance_type}' is not a valid instance type."
        echo "  Verify the type exists in your region with:"
        echo "  aws ec2 describe-instance-types --instance-types ${ec2_type}"
        return 1
    fi

    ACCEL_INSTANCE_TYPE="${instance_type}"
    ACCEL_INSTANCE_COUNT="${instance_count}"
    return 0
}

###########################
## Helm Profile Resolution
###########################

# resolve_helm_profile <instance_type> [instance_count]
# Queries the EC2 API for GPU/EFA specs and sets Helm template variables
# for slurm-values.yaml.template.
# Returns 0 on success, 1 on failure.
resolve_helm_profile() {
    local instance_type="$1"
    local instance_count="${2:-4}"

    if [[ -z "${instance_type}" ]]; then
        echo "Error: --instance-type is required"
        return 1
    fi

    # Validate ml. prefix
    if [[ "${instance_type}" != ml.* ]]; then
        echo "Error: Instance type must start with 'ml.' (got: ${instance_type})"
        return 1
    fi

    # Strip ml. prefix for EC2 API lookup
    local ec2_type="${instance_type#ml.}"

    local info
    if ! info=$(aws ec2 describe-instance-types \
        --instance-types "${ec2_type}" \
        --query 'InstanceTypes[0]' --output json 2>/dev/null); then
        echo "Error: '${instance_type}' is not a valid instance type."
        return 1
    fi

    # Check for Neuron devices (Trainium/Inferentia) — not supported
    local neuron_count
    neuron_count=$(echo "${info}" | jq -r '.NeuronInfo.NeuronDevices[0].Count // 0')
    if [[ "${neuron_count}" != "0" ]]; then
        echo "Error: Neuron/Trainium instances are not currently supported for Slurm"
        echo "  GRES configuration. GPU-based instances are required."
        echo "  Got: ${instance_type} (${neuron_count} Neuron devices detected)"
        return 1
    fi

    # Extract GPU info
    GPU_COUNT=$(echo "${info}" | jq -r '.GpuInfo.Gpus[0].Count // 0')

    # Reject CPU-only instances (no GPU, no accelerator)
    if [[ "${GPU_COUNT}" == "0" ]]; then
        echo "Error: '${instance_type}' has no GPUs. GPU-based instances are"
        echo "  required for the accelerated instance group."
        return 1
    fi

    local gpu_model
    gpu_model=$(echo "${info}" | jq -r '.GpuInfo.Gpus[0].Name // "gpu"')

    # Build GRES string: gpu:<model_lowercase>:<count>
    GPU_GRES="gpu:$(echo "${gpu_model}" | tr '[:upper:]' '[:lower:]'):${GPU_COUNT}"

    # Extract EFA info
    local efa_supported
    efa_supported=$(echo "${info}" | jq -r '.NetworkInfo.EfaSupported // false')
    EFA_COUNT=0
    if [[ "${efa_supported}" == "true" ]]; then
        EFA_COUNT=$(echo "${info}" | jq -r '.NetworkInfo.EfaInfo.MaximumEfaInterfaces // 0')
    fi

    HELM_ACCEL_INSTANCE_TYPE="${instance_type}"
    REPLICAS="${instance_count}"
    MGMT_INSTANCE_TYPE="ml.m5.4xlarge"
    PVC_NAME="fsx-claim"
    return 0
}

###########################
## Training Plan Resolve ##
###########################

# resolve_training_plan <plan_name> <region>
# Validates a SageMaker Training Plan exists, resolves its ARN and AZ ID.
# Sets TRAINING_PLAN_ARN and TRAINING_PLAN_AZ_ID.
# Returns 0 on success, 1 on failure.
resolve_training_plan() {
    local plan_name="$1"
    local region="$2"

    if [[ -z "${plan_name}" ]]; then
        echo "Error: --training-plan requires a plan name"
        return 1
    fi

    echo "Resolving training plan '${plan_name}'..."

    local plan_info
    if ! plan_info=$(aws sagemaker describe-training-plan \
        --training-plan-name "${plan_name}" \
        --region "${region}" 2>&1); then
        echo "Error: Training plan '${plan_name}' not found in region ${region}."
        echo "  ${plan_info}"
        return 1
    fi

    # Extract ARN
    TRAINING_PLAN_ARN=$(echo "${plan_info}" | jq -r '.TrainingPlanArn')
    if [[ -z "${TRAINING_PLAN_ARN}" || "${TRAINING_PLAN_ARN}" == "null" ]]; then
        echo "Error: Could not resolve ARN for training plan '${plan_name}'."
        return 1
    fi

    # Check status
    local status
    status=$(echo "${plan_info}" | jq -r '.Status')
    echo "  ARN: ${TRAINING_PLAN_ARN}"
    echo "  Status: ${status}"

    if [[ "${status}" == "Failed" ]]; then
        echo "Error: Training plan '${plan_name}' has status 'Failed'."
        return 1
    fi
    if [[ "${status}" == "Expired" ]]; then
        echo "  WARNING: Training plan '${plan_name}' has status 'Expired'."
    fi

    # Extract AZ name from first reserved capacity
    local az_name
    az_name=$(echo "${plan_info}" | jq -r \
        '.ReservedCapacitySummaries[0].AvailabilityZone // empty')
    if [[ -z "${az_name}" ]]; then
        echo "Error: Training plan '${plan_name}' has no reserved capacity."
        return 1
    fi

    # Resolve AZ name to AZ ID
    local az_id
    if ! az_id=$(aws ec2 describe-availability-zones \
        --zone-names "${az_name}" \
        --region "${region}" \
        --query 'AvailabilityZones[0].ZoneId' --output text 2>&1); then
        echo "Error: Could not resolve AZ ID for '${az_name}'."
        echo "  ${az_id}"
        return 1
    fi

    TRAINING_PLAN_AZ_ID="${az_id}"
    echo "  AZ: ${az_name} (${TRAINING_PLAN_AZ_ID})"
    return 0
}

###########################
## Prerequisite Checks ####
###########################

# check_command <command_name>
# Verifies that a command exists in PATH.
# Returns 0 if found, 1 if not found.
check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "Error: '$1' is required but not installed."
        return 1
    fi
    return 0
}

###########################
## AZ Validation ##########
###########################

# validate_az_id <az_id> <comma_separated_az_ids>
# Checks if the specified AZ ID exists in the comma-separated list.
# Returns 0 if found, 1 if not found.
validate_az_id() {
    local az_id="$1"
    local az_ids="$2"

    if echo "${az_ids}" | tr ',' '\n' | grep -q "^${az_id}$"; then
        return 0
    else
        return 1
    fi
}

###########################
## CFN Param Resolution ###
###########################

# resolve_cfn_params <params_file> <az_ids> <az_id> <accel_type> <accel_count> [training_plan_arn]
# Runs the jq filter to substitute AZ IDs and instance overrides.
# Outputs resolved JSON to stdout.
# Returns 0 on success, 1 on failure.
resolve_cfn_params() {
    local params_file="$1"
    local az_ids="$2"
    local az_id="$3"
    local accel_type="$4"
    local accel_count="$5"
    local training_plan_arn="${6:-}"

    if [[ ! -f "${params_file}" ]]; then
        echo "Error: Parameters file not found: ${params_file}" >&2
        return 1
    fi

    jq \
        --arg az_ids "${az_ids}" \
        --arg az_id "${az_id}" \
        --arg accel_type "${accel_type}" \
        --argjson accel_count "${accel_count}" \
        --arg training_plan_arn "${training_plan_arn}" \
        '
        map(
            if .ParameterKey == "AvailabilityZoneIds" then
                .ParameterValue = $az_ids
            elif .ParameterKey == "FsxAvailabilityZoneId" then
                .ParameterValue = $az_id
            elif .ParameterKey == "InstanceGroupSettings1" then
                .ParameterValue = (
                    .ParameterValue | fromjson |
                    map(
                        .TargetAvailabilityZoneId = $az_id |
                        if .InstanceGroupName == "accelerated-instance-group-1" then
                            .InstanceType = $accel_type |
                            .InstanceCount = $accel_count |
                            if $training_plan_arn != "" then
                                .TrainingPlanArn = $training_plan_arn
                            else .
                            end
                        else .
                        end
                    ) |
                    tojson
                )
            else .
            end
        )
        ' "${params_file}"
}

###########################
## CFN Template Validation
###########################

# validate_cfn_template <template_url> <resolved_params_file> <region>
# Validates that the S3-hosted CFN template is reachable and cross-checks
# parameter keys in the resolved params file against the template's declared
# parameters.
#
# - Calls aws cloudformation validate-template --template-url
# - Extracts expected parameter keys (and whether they have defaults)
# - Warns about extra params in our file that the template doesn't expect
# - Errors on required params (no default) missing from our file
#
# Returns 0 on success (with possible warnings), 1 on failure.
validate_cfn_template() {
    local template_url="$1"
    local resolved_params_file="$2"
    local region="${3:-us-west-2}"

    if [[ -z "${template_url}" ]]; then
        echo "Error: template_url is required" >&2
        return 1
    fi

    if [[ ! -f "${resolved_params_file}" ]]; then
        echo "Error: Resolved params file not found: ${resolved_params_file}" >&2
        return 1
    fi

    echo "Validating CloudFormation template..."
    echo "  Template URL: ${template_url}"

    # Call validate-template to get parameter metadata
    local validation_output
    if ! validation_output=$(aws cloudformation validate-template \
        --template-url "${template_url}" \
        --region "${region}" 2>&1); then
        echo "Error: Template validation failed:" >&2
        echo "  ${validation_output}" >&2
        return 1
    fi

    echo "  Template syntax: OK"

    # Extract template parameter keys and their default status
    # Parameters with DefaultValue set are optional; those without are required
    local template_keys_with_defaults
    template_keys_with_defaults=$(echo "${validation_output}" | jq -r '
        .Parameters[] |
        .ParameterKey + ":" + (if has("DefaultValue") then "has_default" else "no_default" end)
    ' 2>/dev/null)

    if [[ -z "${template_keys_with_defaults}" ]]; then
        echo "  WARNING: Could not extract parameter metadata from template."
        echo "  Skipping parameter cross-check."
        return 0
    fi

    # Build sets of template keys
    local template_keys
    template_keys=$(echo "${template_keys_with_defaults}" | cut -d: -f1 | sort)

    local required_keys
    required_keys=$(echo "${template_keys_with_defaults}" | \
        grep ':no_default$' | cut -d: -f1 | sort)

    # Extract our resolved params keys
    local our_keys
    our_keys=$(jq -r '.[].ParameterKey' "${resolved_params_file}" | sort)

    # Check for required template params missing from our file
    local missing_required=""
    while IFS= read -r key; do
        [[ -z "${key}" ]] && continue
        if ! echo "${our_keys}" | grep -q "^${key}$"; then
            missing_required="${missing_required}    - ${key}\n"
        fi
    done <<< "${required_keys}"

    if [[ -n "${missing_required}" ]]; then
        echo "" >&2
        echo "Error: Required template parameters missing from params file:" >&2
        printf "%b" "${missing_required}" >&2
        echo "  These parameters have no default value and must be provided." >&2
        return 1
    fi

    # Check for extra params in our file not in the template (warnings only)
    local extra_params=""
    while IFS= read -r key; do
        [[ -z "${key}" ]] && continue
        if ! echo "${template_keys}" | grep -q "^${key}$"; then
            extra_params="${extra_params}    - ${key}\n"
        fi
    done <<< "${our_keys}"

    if [[ -n "${extra_params}" ]]; then
        echo ""
        echo "  WARNING: Parameters in params file not found in template:"
        printf "%b" "${extra_params}"
        echo "  These will be ignored by CloudFormation."
    fi

    # Summary
    local our_count
    our_count=$(echo "${our_keys}" | wc -l | tr -d ' ')
    local template_count
    template_count=$(echo "${template_keys}" | wc -l | tr -d ' ')
    echo "  Parameter cross-check: ${our_count} provided, ${template_count} in template"
    echo "  Validation: OK"

    return 0
}

###########################
## TF Var Resolution ######
###########################

# resolve_tf_vars <target_file> <region> <az_id> <accel_type> <accel_count> [training_plan_arn]
# Patches a tfvars file in-place with region, AZ, and instance overrides.
# Always patches the first instance group's type and count.
# The target file must already exist (copied from source before calling).
# Returns 0 on success, 1 on failure.
resolve_tf_vars() {
    local target_file="$1"
    local region="$2"
    local az_id="$3"
    local accel_type="$4"
    local accel_count="$5"
    local training_plan_arn="${6:-}"

    if [[ ! -f "${target_file}" ]]; then
        echo "Error: tfvars file not found: ${target_file}" >&2
        return 1
    fi

    # Override the aws_region
    sed -i.bak \
        "s|aws_region.*=.*|aws_region            = \"${region}\"|" \
        "${target_file}"

    # Override availability_zone_id in all instance groups
    sed -i.bak \
        "s|availability_zone_id.*=.*|availability_zone_id      = \"${az_id}\",|" \
        "${target_file}"

    # Patch the first instance group's instance_type.
    # Uses awk to replace only the first occurrence of instance_type within
    # the instance_groups block, regardless of its current value.
    awk -v new_type="${accel_type}" '
        /instance_type[[:space:]]*=/ && !type_done {
            sub(/=.*/, "= \"" new_type "\",")
            type_done = 1
        }
        { print }
    ' "${target_file}" > "${target_file}.tmp" && mv "${target_file}.tmp" "${target_file}"

    # Patch the first instance group's instance_count.
    awk -v new_count="${accel_count}" '
        /instance_count[[:space:]]*=/ && !count_done {
            sub(/=.*/, "= " new_count ",")
            count_done = 1
        }
        { print }
    ' "${target_file}" > "${target_file}.tmp" && mv "${target_file}.tmp" "${target_file}"

    # Clean up sed backup files
    rm -f "${target_file}.bak"

    # Inject training_plan_arn into the first instance group if provided
    if [[ -n "${training_plan_arn}" ]]; then
        # Remove existing training_plan_arn if present (idempotent re-run)
        sed -i.bak '/training_plan_arn/d' "${target_file}"
        rm -f "${target_file}.bak"

        awk -v plan_arn="${training_plan_arn}" '
            /lifecycle_script/ && !plan_done {
                print
                print "    training_plan_arn         = \"" plan_arn "\""
                plan_done = 1
                next
            }
            { print }
        ' "${target_file}" > "${target_file}.tmp" \
            && mv "${target_file}.tmp" "${target_file}"
    fi

    return 0
}
