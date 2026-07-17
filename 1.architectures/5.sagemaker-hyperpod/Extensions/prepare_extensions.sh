#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# prepare_extensions.sh
# ---------------------
# Helper to stage and upload SageMaker HyperPod Slurm cluster extensions
# to an S3 bucket. Prints the exact s3:// entrypoint path you paste into
# the HyperPod console under "Custom Setup > Lifecycle configuration >
# None (Run extensions)".
#
# The script:
#   1. Resolves an S3 bucket (existing or newly created).
#   2. Lets you pick which extensions to include (--add-users,
#      --observability, or both).
#   3. Collects any user-supplied inputs (users list, AMP workspace URL).
#   4. Stages the chosen extensions in a temp dir and uploads.
#
# Entrypoint selection:
#   --observability only         -> observability/setup_observability.sh
#                                   (detect-node and run_extensions.sh are
#                                   NOT uploaded)
#   --add-users only             -> run_extensions.sh (detect-node is
#                                   bundled because add_users.sh needs
#                                   nodeinfo.json on the controller)
#   --add-users + --observability -> run_extensions.sh
#
# Usage:
#   prepare_extensions.sh [flags]
#
# Flags:
#   --add-users                 Include the add-users extension
#   --observability             Include the observability extension
#   --users u1,u2,u3            Comma-separated usernames (with --add-users).
#                               UIDs auto-assigned starting at 2001 unless
#                               --uids is given.
#   --uids 2001,2002,2003       Comma-separated UIDs; count must match --users.
#   --users-file <path>         Escape hatch: path to a pre-made
#                               shared_users.txt or shared_users.yaml.
#                               Overrides --users / --uids.
#   --bucket <name>             Use an existing S3 bucket
#   --create-bucket <name>      Create a new S3 bucket with this name
#   --prefix <path>             Object key prefix inside the bucket
#                               (default: hyperpod-extensions)
#   --region <aws-region>       AWS region (default: from AWS CLI config)
#   --amp-url <url>             Prometheus remote_write URL for observability
#   --aws-profile <name>        AWS CLI profile to use (default: current
#                               AWS_PROFILE env or 'default')
#   --dry-run                   Print aws commands instead of executing them
#   --yes, -y                   Skip interactive prompts where possible
#   -h, --help                  Show this help

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PREFIX="hyperpod-extensions"

# ---------------------------------------------------------------------------
# Colors (only when stdout is a TTY)
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'
else
    C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""
fi

info()  { printf "%s[info]%s %s\n"  "$C_BLUE"   "$C_RESET" "$*"; }
ok()    { printf "%s[ ok ]%s %s\n"  "$C_GREEN"  "$C_RESET" "$*"; }
warn()  { printf "%s[warn]%s %s\n"  "$C_YELLOW" "$C_RESET" "$*" >&2; }
err()   { printf "%s[err ]%s %s\n"  "$C_RED"    "$C_RESET" "$*" >&2; }
die()   { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
ENABLE_ADD_USERS="false"
ENABLE_OBSERVABILITY="false"
BUCKET=""
CREATE_BUCKET=""
PREFIX=""
REGION=""
USERS_FILE=""
USERS_CSV=""
UIDS_CSV=""
AMP_URL=""
AWS_PROFILE_ARG=""
DRY_RUN="false"
ASSUME_YES="false"

usage() {
    sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --add-users)       ENABLE_ADD_USERS="true"; shift ;;
        --observability)   ENABLE_OBSERVABILITY="true"; shift ;;
        --bucket)          BUCKET="$2"; shift 2 ;;
        --create-bucket)   CREATE_BUCKET="$2"; shift 2 ;;
        --prefix)          PREFIX="$2"; shift 2 ;;
        --region)          REGION="$2"; shift 2 ;;
        --users-file)      USERS_FILE="$2"; shift 2 ;;
        --users)           USERS_CSV="$2"; shift 2 ;;
        --uids)            UIDS_CSV="$2"; shift 2 ;;
        --amp-url)         AMP_URL="$2"; shift 2 ;;
        --aws-profile)     AWS_PROFILE_ARG="$2"; shift 2 ;;
        --dry-run)         DRY_RUN="true"; shift ;;
        --yes|-y)          ASSUME_YES="true"; shift ;;
        -h|--help)         usage 0 ;;
        *) err "Unknown argument: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# AWS invocation args (profile-aware)
# ---------------------------------------------------------------------------
# Any --profile is applied to every aws call the script makes.
AWS_ARGS=()
if [ -n "$AWS_PROFILE_ARG" ]; then
    AWS_ARGS+=(--profile "$AWS_PROFILE_ARG")
fi

aws_cli() {
    # Wrapper that always includes profile args (used for read-only calls
    # that must run even under --dry-run, e.g. sts, head-bucket, configure get).
    # NOTE: the ${arr[@]+"${arr[@]}"} idiom is required for bash 3.2 (macOS)
    # to safely expand an empty array under `set -u`.
    aws ${AWS_ARGS[@]+"${AWS_ARGS[@]}"} "$@"
}

# ---------------------------------------------------------------------------
# aws wrapper (respects --dry-run) for mutating operations
# ---------------------------------------------------------------------------
run_aws() {
    if [ "$DRY_RUN" = "true" ]; then
        # Show the profile args in the dry-run output so users can verify
        printf "%s[dry ]%s aws %s%s\n" "$C_YELLOW" "$C_RESET" \
            "${AWS_PROFILE_ARG:+--profile $AWS_PROFILE_ARG }" "$*"
        return 0
    fi
    aws ${AWS_ARGS[@]+"${AWS_ARGS[@]}"} "$@"
}

# ---------------------------------------------------------------------------
# S3 bucket name validation (client-side)
# ---------------------------------------------------------------------------
# Applies the naming rules from
# https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
# Returns 0 if valid, 1 if invalid (and echoes the reason to stderr).
validate_bucket_name() {
    local n="$1"

    if [ -z "$n" ]; then
        echo "bucket name is empty" >&2; return 1
    fi
    local len=${#n}
    if [ "$len" -lt 3 ] || [ "$len" -gt 63 ]; then
        echo "must be 3-63 characters (got $len)" >&2; return 1
    fi
    if ! [[ "$n" =~ ^[a-z0-9][a-z0-9.-]*[a-z0-9]$ ]]; then
        echo "must start and end with a lowercase letter or digit; only lowercase, digits, '.', '-' allowed" >&2
        return 1
    fi
    if [[ "$n" == *".."* ]]; then
        echo "must not contain two adjacent periods" >&2; return 1
    fi
    if [[ "$n" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
        echo "must not be formatted as an IP address" >&2; return 1
    fi
    # Reserved prefixes / suffixes
    case "$n" in
        xn--*|sthree-*|amzn-s3-demo-*)
            echo "must not start with reserved prefix (xn--, sthree-, amzn-s3-demo-)" >&2; return 1 ;;
    esac
    case "$n" in
        *-s3alias|*--ol-s3|*.mrap|*--x-s3)
            echo "must not end with reserved suffix (-s3alias, --ol-s3, .mrap, --x-s3)" >&2; return 1 ;;
    esac
    return 0
}

# ---------------------------------------------------------------------------
# Check an existing bucket's status
# ---------------------------------------------------------------------------
# Sets globals BUCKET_STATUS and BUCKET_REGION.
# BUCKET_STATUS values:
#   owned  -> bucket exists and this account owns it
#   other  -> bucket exists but is owned by another account
#   absent -> bucket does not exist
#   error  -> some other AWS error (message left in BUCKET_ERROR)
#
# Requires $CALLER_ACCOUNT to be set (from the sts preflight). Uses
# --expected-bucket-owner so 'exists but not mine' is distinguishable
# from 'exists and mine' -- head-bucket alone no longer distinguishes
# these (as of the 2024 API change that returns metadata for locatable
# buckets regardless of ownership).
check_bucket_status() {
    local bucket="$1"
    BUCKET_STATUS=""; BUCKET_REGION=""; BUCKET_ERROR=""

    if [ "$DRY_RUN" = "true" ]; then
        # Can't call AWS in dry-run; assume owned to allow the flow to complete
        BUCKET_STATUS="owned"; BUCKET_REGION="$REGION"; return 0
    fi

    local out rc
    # Step 1: does the bucket exist at all? (No --expected-bucket-owner here.)
    out="$(aws_cli s3api head-bucket --bucket "$bucket" 2>&1)" || rc=$?
    rc=${rc:-0}

    if [ $rc -ne 0 ]; then
        case "$out" in
            *"Not Found"*|*"(404)"*|*NoSuchBucket*)
                BUCKET_STATUS="absent" ;;
            *InvalidBucketName*|*"(400)"*)
                BUCKET_STATUS="error"; BUCKET_ERROR="invalid bucket name (server-side): $out" ;;
            *"Forbidden"*|*"(403)"*|*AccessDenied*)
                # Some accounts get 403 even for buckets that don't exist in
                # this account -- treat as 'other'.
                BUCKET_STATUS="other" ;;
            *)
                BUCKET_STATUS="error"; BUCKET_ERROR="$out" ;;
        esac
        return 0
    fi

    # Step 2: bucket is locatable -- am I the owner? Use --expected-bucket-owner.
    local owner_out owner_rc=0
    owner_out="$(aws_cli s3api head-bucket \
                    --bucket "$bucket" \
                    --expected-bucket-owner "$CALLER_ACCOUNT" 2>&1)" || owner_rc=$?
    if [ $owner_rc -ne 0 ]; then
        # 403 with expected-bucket-owner -> we are not the owner
        BUCKET_STATUS="other"
        return 0
    fi

    BUCKET_STATUS="owned"
    # Prefer BucketRegion from head-bucket JSON (2024 API), fall back to get-bucket-location
    BUCKET_REGION="$(printf '%s' "$out" | python3 -c \
        'import json,sys
try: print(json.load(sys.stdin).get("BucketRegion",""))
except Exception: pass' 2>/dev/null)"
    if [ -z "$BUCKET_REGION" ]; then
        BUCKET_REGION="$(aws_cli s3api get-bucket-location --bucket "$bucket" \
            --query LocationConstraint --output text 2>/dev/null || true)"
        [ "$BUCKET_REGION" = "None" ] && BUCKET_REGION="us-east-1"
    fi
    return 0
}

confirm() {
    # $1 = prompt, returns 0 on yes
    local ans
    if [ "$ASSUME_YES" = "true" ]; then return 0; fi
    read -r -p "$1 [y/N]: " ans
    [[ "$ans" =~ ^[Yy]([Ee][Ss])?$ ]]
}

prompt() {
    # $1 = prompt, $2 = default (optional). echoes value.
    local ans default="${2:-}"
    if [ -n "$default" ]; then
        read -r -p "$1 [$default]: " ans
        echo "${ans:-$default}"
    else
        read -r -p "$1: " ans
        echo "$ans"
    fi
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
command -v aws >/dev/null 2>&1 || die "aws CLI is not installed or not on PATH"

if [ "$DRY_RUN" != "true" ]; then
    CALLER_JSON="$(aws_cli sts get-caller-identity --output json 2>/dev/null)" \
        || die "AWS credentials not configured or profile '${AWS_PROFILE_ARG:-default}' invalid (aws sts get-caller-identity failed)"
    CALLER_ACCOUNT="$(printf '%s' "$CALLER_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Account"])' 2>/dev/null || echo "unknown")"
    CALLER_ARN="$(printf '%s' "$CALLER_JSON"     | python3 -c 'import json,sys; print(json.load(sys.stdin)["Arn"])'     2>/dev/null || echo "unknown")"
    info "AWS profile: ${AWS_PROFILE_ARG:-<default>}"
    info "AWS account: $CALLER_ACCOUNT"
    info "AWS caller:  $CALLER_ARN"
fi

if [ -n "$BUCKET" ] && [ -n "$CREATE_BUCKET" ]; then
    die "--bucket and --create-bucket are mutually exclusive"
fi

if [ -n "$USERS_CSV$UIDS_CSV$USERS_FILE" ] && [ "$ENABLE_ADD_USERS" != "true" ]; then
    die "--users / --uids / --users-file require --add-users"
fi
if [ -n "$UIDS_CSV" ] && [ -z "$USERS_CSV" ]; then
    die "--uids requires --users"
fi
if [ -n "$USERS_FILE" ] && [ -n "$USERS_CSV$UIDS_CSV" ]; then
    die "--users-file cannot be combined with --users / --uids"
fi

# Ensure the extension source directories exist
for d in detect-node add-users observability; do
    [ -d "$SCRIPT_DIR/$d" ] || die "Missing source directory: $SCRIPT_DIR/$d"
done
[ -f "$SCRIPT_DIR/run_extensions.sh" ] || die "Missing $SCRIPT_DIR/run_extensions.sh"

# ---------------------------------------------------------------------------
# Interactive: which extensions
# ---------------------------------------------------------------------------
if [ "$ENABLE_ADD_USERS" = "false" ] && [ "$ENABLE_OBSERVABILITY" = "false" ] && [ "$ASSUME_YES" != "true" ]; then
    echo
    echo "${C_BOLD}Which extensions do you want to include?${C_RESET}"
    echo "  (detect-node is always included as it's required by run_extensions.sh)"
    if confirm "  Include add-users?"; then ENABLE_ADD_USERS="true"; fi
    if confirm "  Include observability?"; then ENABLE_OBSERVABILITY="true"; fi
fi

if [ "$ENABLE_ADD_USERS" = "false" ] && [ "$ENABLE_OBSERVABILITY" = "false" ]; then
    die "No extensions selected. Pass --add-users and/or --observability."
fi

# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------
if [ -z "$REGION" ]; then
    REGION="$(aws_cli configure get region 2>/dev/null || true)"
fi
if [ -z "$REGION" ] && [ "$ASSUME_YES" != "true" ]; then
    REGION="$(prompt "AWS region" "us-east-1")"
fi
[ -n "$REGION" ] || die "AWS region not set (use --region or configure AWS CLI)"
info "Using region: $REGION"

# ---------------------------------------------------------------------------
# Bucket resolution
# ---------------------------------------------------------------------------
if [ -z "$BUCKET" ] && [ -z "$CREATE_BUCKET" ]; then
    if [ "$ASSUME_YES" = "true" ]; then
        die "No --bucket or --create-bucket provided (required with --yes)"
    fi
    echo
    echo "${C_BOLD}S3 bucket for extension scripts${C_RESET}"
    echo "  1) Use an existing bucket"
    echo "  2) Create a new bucket"
    choice="$(prompt "Choose 1 or 2" "1")"
    case "$choice" in
        1) BUCKET="$(prompt "Existing bucket name")" ;;
        2) CREATE_BUCKET="$(prompt "New bucket name")" ;;
        *) die "Invalid choice: $choice" ;;
    esac
fi

# Client-side name validation for whichever mode we're in
if [ -n "$CREATE_BUCKET" ]; then
    reason="$(validate_bucket_name "$CREATE_BUCKET" 2>&1)" \
        || die "Invalid bucket name '$CREATE_BUCKET': $reason"
elif [ -n "$BUCKET" ]; then
    reason="$(validate_bucket_name "$BUCKET" 2>&1)" \
        || die "Invalid bucket name '$BUCKET': $reason"
fi

if [ -n "$CREATE_BUCKET" ]; then
    info "Requested new bucket: $CREATE_BUCKET (region=$REGION)"
    check_bucket_status "$CREATE_BUCKET"
    case "$BUCKET_STATUS" in
        absent)
            info "Bucket does not exist; creating..."
            if [ "$REGION" = "us-east-1" ]; then
                run_aws s3api create-bucket --bucket "$CREATE_BUCKET" --region "$REGION"
            else
                run_aws s3api create-bucket \
                    --bucket "$CREATE_BUCKET" \
                    --region "$REGION" \
                    --create-bucket-configuration "LocationConstraint=$REGION"
            fi
            # Sensible defaults
            run_aws s3api put-bucket-versioning \
                --bucket "$CREATE_BUCKET" \
                --versioning-configuration Status=Enabled
            run_aws s3api put-public-access-block \
                --bucket "$CREATE_BUCKET" \
                --public-access-block-configuration \
                  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
            ok "Created bucket: $CREATE_BUCKET"
            ;;
        owned)
            if [ -n "$BUCKET_REGION" ] && [ "$BUCKET_REGION" != "$REGION" ]; then
                die "Bucket '$CREATE_BUCKET' already exists in your account but in region '$BUCKET_REGION' (requested: $REGION). Choose a different name or re-run with --region $BUCKET_REGION."
            fi
            warn "Bucket '$CREATE_BUCKET' already exists in this account (region=$BUCKET_REGION); will use it."
            ;;
        other)
            die "Bucket name '$CREATE_BUCKET' is already taken by another AWS account. S3 bucket names are globally unique -- pick a different name (e.g. '${CREATE_BUCKET}-<account-id or suffix>')."
            ;;
        error)
            die "Could not determine status of bucket '$CREATE_BUCKET': $BUCKET_ERROR"
            ;;
    esac
    BUCKET="$CREATE_BUCKET"
else
    info "Using existing bucket: $BUCKET"
    check_bucket_status "$BUCKET"
    case "$BUCKET_STATUS" in
        owned)
            if [ -n "$BUCKET_REGION" ] && [ "$BUCKET_REGION" != "$REGION" ]; then
                die "Bucket '$BUCKET' is in region '$BUCKET_REGION' but --region was '$REGION'. Re-run with --region $BUCKET_REGION (HyperPod requires the extensions bucket to be in the same region as the cluster)."
            fi
            ;;
        absent)
            die "Bucket '$BUCKET' does not exist. Use --create-bucket to create it, or check the name."
            ;;
        other)
            die "Bucket '$BUCKET' exists but is not accessible with this profile/account. Check permissions or use a different bucket."
            ;;
        error)
            die "Could not access bucket '$BUCKET': $BUCKET_ERROR"
            ;;
    esac
fi

# Prefix
if [ -z "$PREFIX" ]; then
    if [ "$ASSUME_YES" = "true" ]; then
        PREFIX="$DEFAULT_PREFIX"
    else
        PREFIX="$(prompt "S3 key prefix" "$DEFAULT_PREFIX")"
    fi
fi
# Strip leading/trailing slashes
PREFIX="${PREFIX#/}"; PREFIX="${PREFIX%/}"
S3_URI="s3://$BUCKET/$PREFIX/"
info "Target: $S3_URI"

# ---------------------------------------------------------------------------
# Decide entrypoint and included dirs based on selection
# ---------------------------------------------------------------------------
# Rules:
#   - only observability      -> entrypoint=setup_observability.sh, no detect-node,
#                                no run_extensions.sh
#   - only add-users          -> entrypoint=run_extensions.sh, include detect-node
#                                (add_users.sh requires nodeinfo.json for Slurm
#                                accounting on the controller)
#   - both                    -> entrypoint=run_extensions.sh, include detect-node
INCLUDE_DETECT_NODE="false"
INCLUDE_RUN_EXTENSIONS="false"
ENTRYPOINT=""

if [ "$ENABLE_ADD_USERS" = "true" ] && [ "$ENABLE_OBSERVABILITY" = "true" ]; then
    INCLUDE_DETECT_NODE="true"
    INCLUDE_RUN_EXTENSIONS="true"
    ENTRYPOINT="run_extensions.sh"
elif [ "$ENABLE_ADD_USERS" = "true" ]; then
    INCLUDE_DETECT_NODE="true"
    INCLUDE_RUN_EXTENSIONS="true"
    ENTRYPOINT="run_extensions.sh"
elif [ "$ENABLE_OBSERVABILITY" = "true" ]; then
    ENTRYPOINT="observability/setup_observability.sh"
else
    die "No extensions selected; nothing to do."
fi

# ---------------------------------------------------------------------------
# Staging directory
# ---------------------------------------------------------------------------
STAGE_DIR="$(mktemp -d -t hyperpod-ext-XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT
info "Staging in: $STAGE_DIR"

if [ "$INCLUDE_DETECT_NODE" = "true" ]; then
    cp -R "$SCRIPT_DIR/detect-node" "$STAGE_DIR/"
fi

# ---------------------------------------------------------------------------
# add-users
# ---------------------------------------------------------------------------
# Builds shared_users.txt from one of three input methods (priority order):
#   1. --users-file <path>   -> copy verbatim (supports .txt and .yaml)
#   2. --users u1,u2,...     -> use provided list; UIDs from --uids or
#                                auto-assigned starting at 2001
#   3. interactive prompt    -> comma-separated usernames + optional UIDs
build_shared_users_txt() {
    # $1 = comma-separated usernames
    # $2 = comma-separated UIDs (may be empty -> auto-assign from 2001)
    # $3 = destination path
    local users_csv="$1" uids_csv="$2" dest="$3"
    local -a users uids
    IFS=',' read -ra users <<<"$users_csv"
    # Trim whitespace
    local i
    for i in "${!users[@]}"; do
        users[$i]="$(printf '%s' "${users[$i]}" | tr -d '[:space:]')"
    done
    # Drop empties
    local -a users_clean=()
    for u in "${users[@]}"; do
        [ -n "$u" ] && users_clean+=("$u")
    done
    users=("${users_clean[@]}")
    [ "${#users[@]}" -gt 0 ] || die "No usernames provided."

    # Validate usernames (POSIX portable-ish: [a-z_][a-z0-9_-]*)
    for u in "${users[@]}"; do
        if ! [[ "$u" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
            die "Invalid username: '$u' (must start with lowercase letter or _; letters/digits/_/- only)"
        fi
    done

    # Detect duplicates
    local -a sorted_unique
    sorted_unique=($(printf '%s\n' "${users[@]}" | sort -u))
    if [ "${#sorted_unique[@]}" -ne "${#users[@]}" ]; then
        die "Duplicate usernames in --users list."
    fi

    if [ -n "$uids_csv" ]; then
        IFS=',' read -ra uids <<<"$uids_csv"
        for i in "${!uids[@]}"; do
            uids[$i]="$(printf '%s' "${uids[$i]}" | tr -d '[:space:]')"
        done
        if [ "${#uids[@]}" -ne "${#users[@]}" ]; then
            die "UID count (${#uids[@]}) does not match username count (${#users[@]})."
        fi
        for uid in "${uids[@]}"; do
            [[ "$uid" =~ ^[0-9]+$ ]] || die "Invalid UID: '$uid' (must be numeric)"
            [ "$uid" -ge 2000 ] || warn "UID $uid is below 2000; may collide with system users."
        done
        # Detect duplicate UIDs
        local -a uids_unique
        uids_unique=($(printf '%s\n' "${uids[@]}" | sort -u))
        if [ "${#uids_unique[@]}" -ne "${#uids[@]}" ]; then
            die "Duplicate UIDs in --uids list."
        fi
    else
        # Auto-assign starting at 2001
        uids=()
        local next=2001
        for _ in "${users[@]}"; do
            uids+=("$next")
            next=$((next+1))
        done
    fi

    : > "$dest"
    for i in "${!users[@]}"; do
        printf "%s,%s,/fsx/%s\n" "${users[$i]}" "${uids[$i]}" "${users[$i]}" >> "$dest"
    done
    ok "Wrote ${#users[@]} user(s) to $(basename "$dest"):"
    while IFS= read -r line; do
        echo "    $line"
    done < "$dest"
}

if [ "$ENABLE_ADD_USERS" = "true" ]; then
    info "Preparing add-users extension"
    cp -R "$SCRIPT_DIR/add-users" "$STAGE_DIR/"
    # Remove the sample files so they don't collide with the real one
    rm -f "$STAGE_DIR/add-users/shared_users_sample.txt" \
          "$STAGE_DIR/add-users/shared_users_sample.yaml"

    if [ -n "$USERS_FILE" ]; then
        [ -f "$USERS_FILE" ] || die "Users file not found: $USERS_FILE"
        case "$USERS_FILE" in
            *.yaml|*.yml) dest="$STAGE_DIR/add-users/shared_users.yaml" ;;
            *)            dest="$STAGE_DIR/add-users/shared_users.txt" ;;
        esac
        cp "$USERS_FILE" "$dest"
        ok "Copied users file -> $(basename "$dest")"
    elif [ -n "$USERS_CSV" ]; then
        build_shared_users_txt "$USERS_CSV" "$UIDS_CSV" "$STAGE_DIR/add-users/shared_users.txt"
    elif [ "$ASSUME_YES" = "true" ]; then
        die "--add-users with --yes requires --users, --uids (optional), or --users-file"
    else
        echo
        echo "${C_BOLD}Add users${C_RESET}"
        read -r -p "  Enter username(s), comma-separated (e.g. 'alice' or 'alice,bob,carol'): " USER_INPUT
        [ -n "$USER_INPUT" ] || die "No usernames entered; aborting."
        read -r -p "  Specify UIDs? (Enter for auto-assign from 2001, or comma-separated UIDs): " UID_INPUT
        build_shared_users_txt "$USER_INPUT" "$UID_INPUT" "$STAGE_DIR/add-users/shared_users.txt"
    fi
fi

# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------
if [ "$ENABLE_OBSERVABILITY" = "true" ]; then
    info "Preparing observability extension"
    cp -R "$SCRIPT_DIR/observability" "$STAGE_DIR/"

    if [ -z "$AMP_URL" ] && [ "$ASSUME_YES" != "true" ]; then
        echo
        echo "${C_BOLD}Observability config${C_RESET}"
        echo "  Enter your Amazon Managed Prometheus remote_write URL."
        echo "  Example: https://aps-workspaces.$REGION.amazonaws.com/workspaces/ws-XXXX/api/v1/remote_write"
        AMP_URL="$(prompt "AMP remote_write URL")"
    fi

    if [ -n "$AMP_URL" ]; then
        cfg="$STAGE_DIR/observability/config.json"
        # Replace the placeholder URL. Use python for robust JSON edit if available;
        # fall back to sed.
        if command -v python3 >/dev/null 2>&1; then
            python3 - "$cfg" "$AMP_URL" <<'PY'
import json, sys
path, url = sys.argv[1], sys.argv[2]
with open(path) as f: cfg = json.load(f)
cfg["prometheus_remote_write_url"] = url
with open(path, "w") as f: json.dump(cfg, f, indent=4); f.write("\n")
PY
        else
            # Escape slashes and & for sed
            esc="$(printf '%s' "$AMP_URL" | sed 's/[\/&]/\\&/g')"
            sed -i.bak "s|\"prometheus_remote_write_url\": \".*\"|\"prometheus_remote_write_url\": \"$esc\"|" "$cfg"
            rm -f "$cfg.bak"
        fi
        ok "Patched observability/config.json"
    else
        warn "Observability config.json still contains placeholder URL; edit before cluster creation."
    fi
fi

# ---------------------------------------------------------------------------
# Patch run_extensions.sh with chosen ENABLE_* flags (only if included)
# ---------------------------------------------------------------------------
# Use a portable sed invocation (macOS + GNU)
sed_i() {
    if sed --version >/dev/null 2>&1; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

if [ "$INCLUDE_RUN_EXTENSIONS" = "true" ]; then
    info "Patching run_extensions.sh flags"
    cp "$SCRIPT_DIR/run_extensions.sh" "$STAGE_DIR/run_extensions.sh"
    sed_i -E "s/^ENABLE_ADD_USERS=\"[^\"]*\"/ENABLE_ADD_USERS=\"$ENABLE_ADD_USERS\"/"           "$STAGE_DIR/run_extensions.sh"
    sed_i -E "s/^ENABLE_OBSERVABILITY=\"[^\"]*\"/ENABLE_OBSERVABILITY=\"$ENABLE_OBSERVABILITY\"/" "$STAGE_DIR/run_extensions.sh"
fi

# Full S3 path to the entrypoint file (what the console asks for)
ENTRYPOINT_S3="s3://$BUCKET/$PREFIX/$ENTRYPOINT"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "${C_BOLD}Ready to upload:${C_RESET}"
echo "  Bucket:            $BUCKET"
echo "  Prefix:            $PREFIX"
echo "  Region:            $REGION"
echo "  AWS profile:       ${AWS_PROFILE_ARG:-<default>}"
echo "  detect-node:       $INCLUDE_DETECT_NODE"
echo "  add-users:         $ENABLE_ADD_USERS"
echo "  observability:     $ENABLE_OBSERVABILITY"
echo "  run_extensions.sh: $INCLUDE_RUN_EXTENSIONS"
echo "  Entrypoint:        $ENTRYPOINT"
echo "  Dry run:           $DRY_RUN"
echo
echo "Staged contents:"
( cd "$STAGE_DIR" && find . -maxdepth 2 -not -path '.' | sort | sed 's|^\./|  |' )
echo

confirm "Proceed with upload?" || die "Aborted before upload."

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
run_aws s3 cp "$STAGE_DIR/" "$S3_URI" --recursive

echo
ok "Upload complete."
echo
echo "${C_BOLD}Paste this into the HyperPod console${C_RESET}"
echo "  Custom setup -> Lifecycle configuration -> None"
echo
echo "  Entrypoint (full S3 path):"
echo "    ${C_GREEN}${ENTRYPOINT_S3}${C_RESET}"
echo
echo "  Extensions bucket URI (if the console asks for it separately):"
echo "    ${C_GREEN}${S3_URI}${C_RESET}"
echo
