---
name: bash-testing
description: Patterns for unit testing bash scripts using bats-core, including AWS CLI mocking, jq/sed/awk testing, and cross-platform portability
---

# Bash Testing with bats-core

## Overview

This skill provides patterns for unit testing bash scripts using
[bats-core](https://github.com/bats-core/bats-core) (Bash Automated Testing
System). It is designed for infrastructure-as-code projects that have bash
scripts calling AWS CLI, jq, sed, terraform, and similar tools.

All patterns below are portable across macOS (BSD) and Linux (GNU).

## Directory Structure

```
project/
  lib/
    <script>_helpers.sh       # Extracted testable functions
  tests/
    test_<script>.bats        # bats test files
    install_bats_libs.sh      # Installs bats-assert + bats-support (run once)
    fixtures/                 # Static test data (JSON, tfvars, YAML)
    helpers/
      mock_aws.bash           # AWS CLI mock function
      setup.bash              # Common bats setup/teardown
    bats/                     # .gitignored — not committed to the repo
      bats-assert/            # bats-assert helper library
      bats-support/           # bats-support helper library
```

**Important:** `tests/bats/` is listed in `.gitignore`. The helper libraries
are cloned locally by `tests/install_bats_libs.sh` and never committed.

## Installation

### 1. Install bats-core

```bash
# macOS
brew install bats-core

# Debian / Ubuntu
sudo apt-get install -y bats

# Fedora / RHEL
sudo dnf install -y bats

# Cross-platform (requires Node.js)
npm install -g bats
```

### 2. Install bats helper libraries

The project includes an install script that clones bats-assert and
bats-support into `tests/bats/`. It is idempotent — safe to run repeatedly.

```bash
bash tests/install_bats_libs.sh
```

This clones the libraries with `--depth 1`, strips `.git/` directories, and
places them under `tests/bats/`. The `.gitignore` entry for `tests/bats/`
prevents them from being committed.

## Running Tests

```bash
# Prerequisites (one-time after clone)
brew install bats-core            # or apt/dnf/npm — see above
bash tests/install_bats_libs.sh

# Run all tests
bats tests/

# Run a specific test file
bats tests/test_deploy.bats

# Verbose output (show test names)
bats --verbose-run tests/test_deploy.bats

# TAP output (for CI integration)
bats --formatter tap tests/
```

## Key Patterns

### 1. Extracting Testable Functions

Split bash scripts into two files:

- **Entry point** (`deploy.sh`): Parses args, orchestrates, calls functions
- **Helpers** (`lib/deploy_helpers.sh`): Pure functions that can be sourced
  and called independently in tests

Functions should:
- Accept inputs as arguments (not rely on global state where possible)
- Output results to stdout (not write to global variables)
- Return meaningful exit codes (0 = success, 1 = failure)
- Be side-effect-free when possible (no network calls in the function
  itself — pass data in, get data out)

### 2. AWS CLI Mocking

Create a bash function named `aws` that intercepts AWS CLI calls and returns
canned responses. Export the function so subshells see it.

```bash
# tests/helpers/mock_aws.bash
mock_aws() {
    aws() {
        case "$*" in
            *"sts get-caller-identity"*)
                echo '{"Account": "123456789012"}'
                ;;
            *"ec2 describe-availability-zones"*)
                # Return tab-separated AZ IDs (mimics --output text)
                printf "usw2-az1\tusw2-az2\tusw2-az3\tusw2-az4"
                ;;
            *"cloudformation create-stack"*)
                echo '{"StackId": "arn:aws:cloudformation:us-west-2:123:stack/test/guid"}'
                ;;
            *"cloudformation wait"*)
                return 0
                ;;
            *"cloudformation describe-stacks"*)
                echo "mock-value"
                ;;
            *)
                echo "UNMOCKED AWS CALL: aws $*" >&2
                return 1
                ;;
        esac
    }
    export -f aws
}
```

To customize responses per test, override the function inside the test:

```bash
@test "handles empty AZ list" {
    aws() {
        case "$*" in
            *"ec2 describe-availability-zones"*) printf "" ;;
            *) echo "UNMOCKED: $*" >&2; return 1 ;;
        esac
    }
    export -f aws

    run resolve_az_ids "us-west-2"
    assert_failure
}
```

### 3. Testing jq Transformations

Run the jq filter directly against fixture files. Assert on specific JSON
fields in the output.

```bash
@test "jq substitutes AvailabilityZoneIds" {
    result=$(resolve_cfn_params \
        "$FIXTURE_DIR/params.json" \
        "use1-az1,use1-az2,use1-az3" \
        "use1-az2" \
        "ml.g5.8xlarge" \
        4)

    az_ids=$(echo "$result" | jq -r '
        .[] | select(.ParameterKey == "AvailabilityZoneIds") |
        .ParameterValue')

    assert_equal "$az_ids" "use1-az1,use1-az2,use1-az3"
}
```

### 4. Testing sed / awk Transformations

Copy the fixture file to a temp directory, run the function, then assert on
the output file contents.

```bash
@test "sed overrides aws_region" {
    local target="$TEST_TEMP_DIR/custom.tfvars"
    cp "$FIXTURE_DIR/custom.tfvars" "$target"

    resolve_tf_vars "$target" "us-east-1" "use1-az2" "ml.g5.8xlarge" 4 "g5"

    run grep 'aws_region' "$target"
    assert_output --partial 'us-east-1'
}
```

**Portability warning — sed "first occurrence only":**

GNU sed supports `0,/pattern/s||replacement|` to replace only the first
match, but macOS (BSD) sed silently ignores the `0,` address. This means
the substitution does nothing on macOS with no error.

Use awk instead for portable first-occurrence replacement:

```bash
# PORTABLE: works on both macOS and Linux
awk -v new_val="replacement" '
    /pattern/ && !done { sub(/pattern/, new_val); done = 1 }
    { print }
' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
```

**Portability note — sed in-place editing:**

- `sed -i.bak 's/old/new/' file` works on both macOS and Linux. The `.bak`
  extension creates a backup file on both platforms.
- `sed -i '' 's/old/new/' file` is macOS-only (GNU sed treats `''` as the
  backup extension, not "no backup"). Avoid this form.
- Always clean up `.bak` files after: `rm -f "$file.bak"`

### 5. Testing Argument Parsing

For scripts that use `set -euo pipefail`, run them as subprocesses (not
sourced) to test argument validation and exit codes.

```bash
@test "fails when --instance-type is missing" {
    run bash "$PROJECT_DIR/deploy.sh" --infra cfn
    assert_failure
    assert_output --partial "Error: --instance-type is required"
}

@test "--help exits 0 and prints usage" {
    run bash "$PROJECT_DIR/deploy.sh" --help
    assert_success
    assert_output --partial "Usage:"
}
```

### 6. Testing Prerequisite Checks

Temporarily modify PATH to hide commands:

```bash
@test "fails when jq is missing for cfn infra" {
    local fake_bin="$TEST_TEMP_DIR/bin"
    mkdir -p "$fake_bin"
    ln -s "$(which bash)" "$fake_bin/bash"
    ln -s "$(which aws)" "$fake_bin/aws"

    PATH="$fake_bin" run check_command "jq"
    assert_failure
}
```

### 7. Common setup.bash Pattern

```bash
# tests/helpers/setup.bash

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

    # Load AWS mock
    source "${PROJECT_DIR}/tests/helpers/mock_aws.bash"
    mock_aws
}

teardown() {
    # Clean up temp directory
    if [[ -n "${TEST_TEMP_DIR}" && -d "${TEST_TEMP_DIR}" ]]; then
        rm -rf "${TEST_TEMP_DIR}"
    fi
}
```

### 8. install_bats_libs.sh Pattern

```bash
#!/bin/bash
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

echo "Installing bats helper libraries..."
install_lib "bats-support" "https://github.com/bats-core/bats-support"
install_lib "bats-assert" "https://github.com/bats-core/bats-assert"
echo "Done. You can now run: bats tests/test_deploy.bats"
```

## Conventions

- Test file naming: `test_<script-name>.bats`
- Test names should be descriptive: `@test "resolve_cfn_params: p5 overrides
  accelerated group instance type"`
- Group related tests with comment section headers
- Use `run` to capture command output and exit status
- Use `assert_success`, `assert_failure`, `assert_output`, `assert_equal`
  from bats-assert
- Each test should be independent (no shared state between tests)
- Use `$TEST_TEMP_DIR` for any file writes (cleaned up in teardown)
- Fixture files in `tests/fixtures/` are independent copies, not symlinks
- Fixture files are read-only — copy to temp dir before modifying
- `tests/bats/` is in `.gitignore` — never committed to the repo
- Use `tests/install_bats_libs.sh` to install bats helper libraries

## Cross-Platform Notes

- **sed in-place**: Always use `sed -i.bak` (not `sed -i ''`). Clean up
  `.bak` files after.
- **sed first-occurrence**: Do not use `0,/pattern/` — this is GNU-only and
  silently fails on macOS. Use awk instead (see section 4).
- **awk**: POSIX awk is available on both macOS and Linux. The patterns in
  this skill use only POSIX awk features.
- **mktemp**: `mktemp -d` works on both macOS and Linux.
- **bats-core install**: Use `brew` on macOS, `apt`/`dnf` on Linux, or
  `npm install -g bats` cross-platform.
