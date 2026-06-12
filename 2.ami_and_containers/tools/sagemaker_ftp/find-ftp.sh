#!/usr/bin/env bash
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# find-ftp.sh
#
# Find available Amazon SageMaker Flexible Training Plan (FTP) offerings
# for a SageMaker HyperPod cluster (Slurm or EKS) or for SageMaker training
# jobs, scoped to a specific Availability Zone. Sweeps common reservation
# durations and recommends the offering with the lowest effective
# $/instance/hour.
#
# -----------------------------------------------------------------------------
# PREREQUISITES
# -----------------------------------------------------------------------------
#   1. AWS CLI v2 installed and on PATH
#        https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
#   2. jq installed and on PATH
#        macOS:  brew install jq
#        Linux:  sudo apt-get install jq   (or your package manager equivalent)
#   3. AWS credentials configured (default profile, named profile, or env vars)
#        aws configure                  # default profile
#        aws configure --profile NAME   # named profile (use with --profile)
#   4. IAM permission on the calling principal:
#        sagemaker:SearchTrainingPlanOfferings
#      To later purchase a plan you also need:
#        sagemaker:CreateTrainingPlan
#   5. (Optional) Sufficient SageMaker reserved-capacity quota for the
#      requested instance type. If the requested instance count exceeds your
#      quota, the API returns ResourceLimitExceeded for that duration.
#
# This script applies to both HyperPod Slurm and HyperPod EKS clusters; the
# training plan target resource is the same (`hyperpod-cluster`) in both cases.
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

# Default durations (hours) to probe. SearchTrainingPlanOfferings filters
# strictly on --duration-hours, so we sweep common reservation lengths.
# Override via --durations.
DEFAULT_DURATIONS=(24 48 72 168 336 720)

# Default target resource. Override via --target.
# Valid values: hyperpod-cluster (default), training-job.
DEFAULT_TARGET_RESOURCE="hyperpod-cluster"

usage() {
  cat <<EOF
$SCRIPT_NAME — Find available SageMaker Flexible Training Plan (FTP)
offerings in a specific Availability Zone, and recommend the cheapest option.

Supports both HyperPod clusters (Slurm or EKS) and SageMaker training jobs
via the --target flag. Defaults to hyperpod-cluster.

USAGE:
  $SCRIPT_NAME --az <AZ> --instance-type <ML_INSTANCE_TYPE> \\
               --count <N> --region <REGION> \\
               [--start-after <ISO8601_UTC>] [--profile <AWS_PROFILE>] \\
               [--target hyperpod-cluster|training-job] \\
               [--durations <H1,H2,...>] [--json]

REQUIRED ARGUMENTS:
  --az              Availability Zone name, e.g. us-west-2b
  --instance-type   SageMaker instance type with ml. prefix, e.g. ml.p5.48xlarge
  --count           Number of instances to reserve (integer > 0)
  --region          AWS region, e.g. us-west-2

OPTIONAL ARGUMENTS:
  --start-after     Earliest acceptable start time, ISO-8601 UTC.
                    Default: now (e.g. 2026-06-15T00:00:00Z)
  --profile         Named AWS CLI profile to use for credentials.
                    Default: whatever the AWS CLI resolves (env vars,
                    default profile, instance role, etc.)
  --target          Target resource for the training plan. One of:
                      hyperpod-cluster   (default; HyperPod Slurm or EKS)
                      training-job       (SageMaker training jobs)
  --durations       Comma-separated list of durations (in hours) to probe.
                    Default: ${DEFAULT_DURATIONS[*]// /,}
                    Example: --durations 24,168,720
  --json            Emit machine-readable JSON to stdout instead of the
                    human-readable table. Status messages still go to stderr.
  -h, --help        Show this help message and exit.

WHAT IT DOES:
  1. Calls aws sagemaker search-training-plan-offerings for each duration
     with --target-resources <TARGET>.
  2. Filters offerings to the Availability Zone you specified.
  3. Computes effective \$/instance/hour for each offering.
  4. Prints all matches as a table (or JSON with --json).
  5. Recommends the lowest \$/instance/hour offering and prints a ready-to-run
     create-training-plan command for it (text mode only).

EXAMPLES:
  # Search for 2x p5.48xlarge for a HyperPod cluster (default target)
  $SCRIPT_NAME --az us-west-2b --instance-type ml.p5.48xlarge \\
               --count 2 --region us-west-2 \\
               --start-after 2026-06-15T00:00:00Z

  # Same query, but for SageMaker training jobs instead of HyperPod
  $SCRIPT_NAME --az us-west-2b --instance-type ml.p5.48xlarge \\
               --count 2 --region us-west-2 \\
               --start-after 2026-06-15T00:00:00Z \\
               --target training-job

  # Use a named AWS profile and custom durations
  $SCRIPT_NAME --az us-west-2b --instance-type ml.p6-b200.48xlarge \\
               --count 2 --region us-west-2 \\
               --start-after 2026-06-15T00:00:00Z \\
               --profile my-profile \\
               --durations 24,48,168

  # Machine-readable JSON output for piping into other tools
  $SCRIPT_NAME --az us-west-2b --instance-type ml.p5.48xlarge \\
               --count 2 --region us-west-2 --json | jq '.offerings[0]'

EXIT CODES:
  0  Search completed (regardless of whether offerings were found).
  1  Invalid arguments or missing prerequisites.

NOTES:
  * SearchTrainingPlanOfferings only returns offerings whose duration matches
    --duration-hours exactly, which is why this script sweeps several values.
  * Empty results mean no capacity currently matches your criteria; try a
    different AZ, instance count, region, or start window.
  * create-training-plan charges the upfront fee immediately and is
    NON-REFUNDABLE. This script never purchases on your behalf — it only
    prints the command for you to run.
EOF
}

# ---------- argument parsing ----------
AZ=""
INSTANCE_TYPE=""
INSTANCE_COUNT=""
REGION=""
START_AFTER=""
AWS_PROFILE_ARG=""
DURATIONS_ARG=""
TARGET_RESOURCE="$DEFAULT_TARGET_RESOURCE"
JSON_OUT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --az)             AZ="$2"; shift 2 ;;
    --instance-type)  INSTANCE_TYPE="$2"; shift 2 ;;
    --count)          INSTANCE_COUNT="$2"; shift 2 ;;
    --region)         REGION="$2"; shift 2 ;;
    --start-after)    START_AFTER="$2"; shift 2 ;;
    --profile)        AWS_PROFILE_ARG="$2"; shift 2 ;;
    --target)         TARGET_RESOURCE="$2"; shift 2 ;;
    --durations)      DURATIONS_ARG="$2"; shift 2 ;;
    --json)           JSON_OUT=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      echo "Run '$SCRIPT_NAME --help' for usage." >&2
      exit 1
      ;;
  esac
done

# ---------- validation ----------
missing=()
[[ -z "$AZ" ]]             && missing+=("--az")
[[ -z "$INSTANCE_TYPE" ]]  && missing+=("--instance-type")
[[ -z "$INSTANCE_COUNT" ]] && missing+=("--count")
[[ -z "$REGION" ]]         && missing+=("--region")

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: missing required argument(s): ${missing[*]}" >&2
  echo "Run '$SCRIPT_NAME --help' for usage." >&2
  exit 1
fi

if ! [[ "$INSTANCE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --count must be a positive integer (got: $INSTANCE_COUNT)" >&2
  exit 1
fi

case "$TARGET_RESOURCE" in
  hyperpod-cluster|training-job) ;;
  *)
    echo "ERROR: --target must be 'hyperpod-cluster' or 'training-job' (got: $TARGET_RESOURCE)" >&2
    exit 1
    ;;
esac

# Default start time to "now" (UTC) if not provided.
if [[ -z "$START_AFTER" ]]; then
  START_AFTER="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

# Resolve durations.
if [[ -n "$DURATIONS_ARG" ]]; then
  IFS=',' read -r -a DURATIONS <<<"$DURATIONS_ARG"
  for d in "${DURATIONS[@]}"; do
    if ! [[ "$d" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: --durations must be a comma-separated list of positive integers (got: $DURATIONS_ARG)" >&2
      exit 1
    fi
  done
else
  DURATIONS=("${DEFAULT_DURATIONS[@]}")
fi

# ---------- prerequisite checks ----------
command -v aws >/dev/null 2>&1 || {
  echo "ERROR: aws CLI not found on PATH. Install AWS CLI v2." >&2; exit 1; }
command -v jq  >/dev/null 2>&1 || {
  echo "ERROR: jq not found on PATH. Install jq (e.g. 'brew install jq')." >&2; exit 1; }

# Build optional profile flag for aws calls.
AWS_PROFILE_FLAG=()
if [[ -n "$AWS_PROFILE_ARG" ]]; then
  AWS_PROFILE_FLAG=(--profile "$AWS_PROFILE_ARG")
fi

# Helper: log to stderr (so JSON on stdout stays clean in --json mode).
log() { echo "$@" >&2; }

# ---------- run ----------
log "Searching SageMaker FTP offerings:"
log "  AZ:             $AZ"
log "  Instance type:  $INSTANCE_TYPE"
log "  Count:          $INSTANCE_COUNT"
log "  Region:         $REGION"
log "  Start after:    $START_AFTER"
log "  Target:         $TARGET_RESOURCE"
log "  Profile:        ${AWS_PROFILE_ARG:-<default>}"
log "  Durations (h):  ${DURATIONS[*]}"
log ""

ALL_OFFERINGS="[]"

for DUR in "${DURATIONS[@]}"; do
  if ! RESP=$(aws "${AWS_PROFILE_FLAG[@]}" sagemaker search-training-plan-offerings \
        --region "$REGION" \
        --instance-type "$INSTANCE_TYPE" \
        --instance-count "$INSTANCE_COUNT" \
        --start-time-after "$START_AFTER" \
        --duration-hours "$DUR" \
        --target-resources "$TARGET_RESOURCE" \
        --output json 2>/dev/null); then
    # API errored (e.g. ResourceLimitExceeded for that duration). Skip silently.
    continue
  fi

  MATCHES=$(echo "$RESP" | jq --arg az "$AZ" --argjson count "$INSTANCE_COUNT" '
    [.TrainingPlanOfferings[]
     | select(any(.ReservedCapacityOfferings[]; .AvailabilityZone == $az))
     | {
         id: .TrainingPlanOfferingId,
         duration_h: .DurationHours,
         upfront: (.UpfrontFee | tonumber),
         currency: .CurrencyCode,
         az: .ReservedCapacityOfferings[0].AvailabilityZone,
         start: .ReservedCapacityOfferings[0].StartTime,
         end:   .ReservedCapacityOfferings[0].EndTime,
         per_inst_hr: ((.UpfrontFee | tonumber) / (.DurationHours * $count))
       }
    ]')

  ALL_OFFERINGS=$(jq -n --argjson a "$ALL_OFFERINGS" --argjson b "$MATCHES" '$a + $b')
done

TOTAL=$(echo "$ALL_OFFERINGS" | jq 'length')

# ---------- JSON output mode ----------
if [[ "$JSON_OUT" -eq 1 ]]; then
  jq -n \
    --arg az "$AZ" \
    --arg instance_type "$INSTANCE_TYPE" \
    --argjson count "$INSTANCE_COUNT" \
    --arg region "$REGION" \
    --arg start_after "$START_AFTER" \
    --arg target "$TARGET_RESOURCE" \
    --argjson durations "$(printf '%s\n' "${DURATIONS[@]}" | jq -R 'tonumber' | jq -s '.')" \
    --argjson offerings "$ALL_OFFERINGS" \
    '{
       query: {
         availability_zone: $az,
         instance_type: $instance_type,
         instance_count: $count,
         region: $region,
         start_after: $start_after,
         target_resources: $target,
         durations_hours: $durations
       },
       count: ($offerings | length),
       offerings: ($offerings | sort_by(.per_inst_hr)),
       recommendation: (if ($offerings | length) > 0
                        then ($offerings | sort_by(.per_inst_hr) | .[0])
                        else null
                        end)
     }'
  exit 0
fi

# ---------- text output mode ----------
if [[ "$TOTAL" -eq 0 ]]; then
  echo "No offerings found in AZ $AZ for $INSTANCE_TYPE x$INSTANCE_COUNT"
  echo "across the probed durations (${DURATIONS[*]} h)."
  echo
  echo "Suggestions:"
  echo "  * Try a different AZ in the same region."
  echo "  * Try a smaller --count (your account quota may be limiting results)."
  echo "  * Try a later --start-after."
  echo "  * Try a different --region."
  echo "  * Try --durations with other values (e.g. 96,240,480)."
  exit 0
fi

echo "Found $TOTAL matching offering(s) in $AZ:"
echo
printf "%-50s %8s %12s %10s   %-20s %-20s\n" \
  "OFFERING_ID" "DUR(h)" "UPFRONT" "\$/inst/hr" "START (UTC)" "END (UTC)"
printf "%-50s %8s %12s %10s   %-20s %-20s\n" \
  "$(printf '%.0s-' {1..50})" "------" "------------" "----------" "--------------------" "--------------------"

echo "$ALL_OFFERINGS" | jq -r '.[] |
  [
    .id,
    (.duration_h | tostring),
    (.upfront | tostring),
    ((.per_inst_hr * 100 | floor) / 100 | tostring),
    (.start | strftime("%Y-%m-%d %H:%M")),
    (.end   | strftime("%Y-%m-%d %H:%M"))
  ] | @tsv' | \
while IFS=$'\t' read -r id dur upfront rate start end; do
  printf "%-50s %8s %12s %10s   %-20s %-20s\n" \
    "${id:0:48}.." "$dur" "$upfront" "$rate" "$start" "$end"
done

echo
echo "Recommendation (lowest \$/instance/hour):"
echo "$ALL_OFFERINGS" | jq -r '
  sort_by(.per_inst_hr) | .[0] |
  "  Offering ID:  " + .id,
  "  Duration:     " + (.duration_h | tostring) + " h",
  "  AZ:           " + .az,
  "  Upfront fee:  " + (.upfront | tostring) + " " + .currency,
  "  Effective:    $" + (((.per_inst_hr * 100 | floor) / 100) | tostring) + " / instance / hour",
  "  Window:       " + (.start | strftime("%Y-%m-%d %H:%M UTC")) + " -> " + (.end | strftime("%Y-%m-%d %H:%M UTC"))
'

echo
echo "To purchase this offering (NON-REFUNDABLE; charges the upfront fee immediately):"
BEST_ID=$(echo "$ALL_OFFERINGS" | jq -r 'sort_by(.per_inst_hr) | .[0].id')
echo "  aws sagemaker create-training-plan \\"
echo "    --region $REGION \\"
if [[ -n "$AWS_PROFILE_ARG" ]]; then
  echo "    --profile $AWS_PROFILE_ARG \\"
fi
echo "    --training-plan-name <YOUR_PLAN_NAME> \\"
echo "    --training-plan-offering-id $BEST_ID"
