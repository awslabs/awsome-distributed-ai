#!/usr/bin/env bash
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# efa-nccl-doctor.sh
#
# Lint Dockerfiles that build EFA-enabled distributed-training containers, and
# report version drift and known-bad combinations BEFORE you spend 20 minutes
# on a build and a multi-node allocation discovering the problem at runtime.
#
# WHY THIS EXISTS
# -----------------------------------------------------------------------------
# Building a container that actually uses EFA requires getting several
# independent things right at once:
#
#   * efa-installer version
#   * aws-ofi-nccl version (and its confusing tag naming: 1.12.1-aws vs
#     v1.13.2-aws vs v1.19.0 -- upstream dropped the -aws suffix at v1.14.0)
#   * NCCL version compatibility with the above
#   * the three container-only efa_installer.sh flags
#     (--skip-kmod --no-verify --skip-limit-conf) that MUST NOT be used on a host
#   * removing the bundled HPC-X OMPI/NCCL plugin that otherwise shadows EFA
#   * LD_LIBRARY_PATH ordering so libfabric resolves to /opt/amazon/efa/lib
#
# Every one of these fails SILENTLY in the sense that matters: the container
# builds fine, the job launches fine, and NCCL quietly falls back to TCP. You
# discover it as "why is my 32-node run slower than 8 nodes".
#
# This linter encodes those rules as static checks over Dockerfiles.
#
# -----------------------------------------------------------------------------
# PREREQUISITES
# -----------------------------------------------------------------------------
#   1. bash 4+, awk, sed, grep, sort  (present on AL2023, Ubuntu, macOS w/ brew bash)
#   2. curl  -- only for --check-latest (online version resolution)
#   3. jq    -- only for --json or --check-latest
#
# The default (offline) mode needs no network and no AWS credentials.
# -----------------------------------------------------------------------------

set -uo pipefail

SCRIPT_NAME="$(basename "$0")"
APP_VERSION="1.0 (2026 Aug)"

# -----------------------------------------------------------------------------
# Known compatibility rules.
#
# Sources:
#   * 1.architectures/efa-cheatsheet.md in this repository
#   * 2.ami_and_containers/containers/pytorch/0.nvcr-pytorch-aws.dockerfile
#   * https://github.com/aws/aws-ofi-nccl/releases
#
# These are FLOOR values, not "latest". The linter never tells you to chase the
# newest release -- it tells you when a combination is known to misbehave.
# -----------------------------------------------------------------------------

# efa-installer below this + nccl >= 2.19.0 requires FI_EFA_SET_CUDA_SYNC_MEMOPS=0
# or NCCL fails with "register_rail_mr_buffer ... Unable to register memory".
EFA_CUDA_SYNC_MEMOPS_FLOOR="1.29.1"

# aws-ofi-nccl dropped the "-aws" suffix at v1.14.0. Tags below that need it;
# tags at or above must NOT have it, or the git clone 404s.
OFI_NCCL_SUFFIX_DROP="1.14.0"

# Container-only efa_installer.sh flags. Correct INSIDE a Dockerfile,
# actively wrong on a host.
CONTAINER_ONLY_FLAGS=("--skip-kmod" "--no-verify")

# ---------- output helpers ----------
C_RED=""; C_YEL=""; C_GRN=""; C_DIM=""; C_RST=""
if [[ -t 1 ]] && [[ "${NO_COLOR:-}" == "" ]]; then
  C_RED=$'\033[31m'; C_YEL=$'\033[33m'; C_GRN=$'\033[32m'
  C_DIM=$'\033[2m';  C_RST=$'\033[0m'
fi

ERRORS=0
WARNINGS=0
FINDINGS_JSON="[]"

log()  { echo "$@" >&2; }
emit() { [[ "$JSON_OUT" -eq 0 ]] && echo "$@"; }

# record a finding: level, file, line, code, message, remedy
finding() {
  local level="$1" file="$2" line="$3" code="$4" msg="$5" remedy="${6:-}"

  # Honor an inline opt-out pragma for this code.
  if is_suppressed "$file" "$code"; then
    [[ "$JSON_OUT" -eq 0 ]] && printf '  %bskip %s [%s] suppressed by pragma%b\n' \
      "$C_DIM" "$(basename "$file")" "$code" "$C_RST"
    return
  fi
  case "$level" in
    ERROR) ERRORS=$((ERRORS + 1));   local tag="${C_RED}ERROR${C_RST}"  ;;
    WARN)  WARNINGS=$((WARNINGS + 1)); local tag="${C_YEL}WARN ${C_RST}" ;;
    INFO)  local tag="${C_DIM}INFO ${C_RST}" ;;
  esac

  if [[ "$JSON_OUT" -eq 1 ]]; then
    FINDINGS_JSON=$(jq -n \
      --argjson acc "$FINDINGS_JSON" \
      --arg level "$level" --arg file "$file" --argjson line "${line:-0}" \
      --arg code "$code" --arg msg "$msg" --arg remedy "$remedy" \
      '$acc + [{level:$level, file:$file, line:$line, code:$code,
                message:$msg, remedy:$remedy}]')
  else
    printf '  %b %s:%s [%s]\n' "$tag" "$(basename "$file")" "${line:-–}" "$code"
    printf '        %s\n' "$msg"
    [[ -n "$remedy" ]] && printf '        %b-> %s%b\n' "$C_DIM" "$remedy" "$C_RST"
  fi
}

# semantic version compare: returns 0 if $1 < $2
ver_lt() {
  [[ "$1" == "$2" ]] && return 1
  local lower
  lower="$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)"
  [[ "$lower" == "$1" ]]
}

usage() {
  cat <<EOF
$SCRIPT_NAME - $APP_VERSION

Lint EFA/NCCL container recipes for version drift and known-bad combinations.

USAGE:
  $SCRIPT_NAME [OPTIONS] <PATH> [<PATH>...]

ARGUMENTS:
  <PATH>              A Dockerfile, or a directory to scan recursively for
                      files matching Dockerfile* or *.dockerfile.

OPTIONS:
  --check-latest      Resolve the newest aws-ofi-nccl release from GitHub and
                      report how far behind each file is. Requires network
                      access and jq. Informational only; never fails the run.
  --inventory         Print a version inventory table across all scanned files
                      instead of per-file findings. Useful for spotting drift
                      across a repository at a glance.
  --strict            Treat warnings as errors (exit 1 if any WARN is emitted).
  --json              Emit machine-readable JSON on stdout. Requires jq.
  --no-color          Disable ANSI color (also honors NO_COLOR env var).
  -h, --help          Show this help and exit.
  -V, --version       Print version and exit.

CHECKS PERFORMED:
  EFA001  efa-installer version not pinned (uses latest / unset)
  EFA002  efa-installer < ${EFA_CUDA_SYNC_MEMOPS_FLOOR} without FI_EFA_SET_CUDA_SYNC_MEMOPS=0
  EFA003  container-only installer flags missing (--skip-kmod / --no-verify)
  EFA004  --skip-limit-conf missing (redundant but conventional in containers)
  NCCL001 aws-ofi-nccl version not pinned
  NCCL002 aws-ofi-nccl tag suffix wrong for its version (-aws dropped at v${OFI_NCCL_SUFFIX_DROP})
  NCCL003 HPC-X bundled OMPI/NCCL plugin not removed (shadows EFA)
  NCCL004 aws-ofi-nccl built without --with-libfabric pointing at /opt/amazon/efa
  ENV001  LD_LIBRARY_PATH does not include /opt/amazon/efa/lib
  ENV002  OPAL_PREFIX not cleared after removing bundled OMPI
  PIN001  base image uses a floating tag (:latest or no tag)

EXIT CODES:
  0  No errors (warnings allowed unless --strict).
  1  One or more errors, or invalid usage.

EXAMPLES:
  # Lint a single Dockerfile
  $SCRIPT_NAME 2.ami_and_containers/containers/pytorch/0.nvcr-pytorch-aws.dockerfile

  # Scan an entire repository and show the drift inventory
  $SCRIPT_NAME --inventory .

  # CI gate: fail the build on any error
  $SCRIPT_NAME --strict 3.test_cases/

  # Machine-readable, piped into another tool
  $SCRIPT_NAME --json . | jq '.findings[] | select(.level=="ERROR")'

NOTES:
  * The default mode is fully offline. --check-latest is the only network call.
  * This linter is intentionally conservative: it flags patterns that are known
    to cause silent TCP fallback, not stylistic preferences.
EOF
}

# -----------------------------------------------------------------------------
# checks
# -----------------------------------------------------------------------------

# Respect an inline opt-out, in the spirit of "# shellcheck disable=SCxxxx".
# Some deviations are deliberate and verified -- e.g. nemo:26.04.01 links
# torch against /opt/hpcx/ompi/lib, so removing HPC-X breaks `import torch`.
# A linter that cannot be silenced with a documented reason gets ignored
# wholesale, which is worse than a linter with escape hatches.
#
#   # efa-nccl-doctor: disable=NCCL003 reason=torch links libmpi.so.40 from hpcx
is_suppressed() {
  local file="$1" code="$2"
  grep -qE "efa-nccl-doctor:[[:space:]]*disable=[^[:space:]]*${code}" "$file" 2>/dev/null
}

# Strip comment lines and blank lines. Dockerfiles in this repo carry long
# explanatory comment blocks that frequently MENTION images and flags they
# deliberately do NOT use ("we avoid nvcr.io/nvidia/pytorch because..."), so
# matching raw file text produces false positives. All content checks below
# operate on the decommented body.
decomment() {
  sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$1"
}

# Match a pattern against an already-decommented body held in a variable.
#
# We deliberately do NOT write `decomment "$f" | grep -q ...`. A pipeline whose
# right-hand side is `grep -q` can consume a file descriptor left open by the
# caller (FILES is populated via `while read < <(find ...)`), which made the
# same file lint differently depending on whether it was passed directly or
# discovered by a directory scan. Reading the body once into a variable and
# matching with a here-string removes the pipeline, and with it the ambiguity.
body_has() {
  local body="$1" pattern="$2"
  grep -qE "$pattern" <<<"$body"
}

# extract "VAR=value" or "VAR value" from ENV/ARG lines
extract_var() {
  local file="$1" var="$2"
  grep -oP "^\s*(ENV|ARG)\s+${var}[= ]+\K[^\s\\\\]+" "$file" 2>/dev/null | tail -1
}

line_of() {
  local file="$1" pattern="$2"
  grep -nE "$pattern" "$file" 2>/dev/null | head -1 | cut -d: -f1
}

lint_file() {
  local file="$1"
  local efa_ver ofi_ver base_img body

  emit "${C_DIM}--- ${file}${C_RST}"

  # Decomment ONCE, up front. Every content check below matches against this
  # body rather than the raw file, because these Dockerfiles carry long
  # explanatory comment blocks that routinely mention images, flags and
  # packages they deliberately do NOT use.
  body="$(decomment "$file")"

  # Only lint files that actually install EFA or build aws-ofi-nccl. A
  # Dockerfile with no EFA is not broken; it is out of scope. Gating on the
  # decommented body means a file that merely *discusses* EFA in a comment
  # is correctly skipped.
  if ! body_has "$body" "efa.installer|efa_installer|aws-ofi-nccl"; then
    emit "  ${C_DIM}skipped (no EFA / aws-ofi-nccl references)${C_RST}"
    return
  fi

  # ---------- EFA installer version ----------
  efa_ver="$(extract_var "$file" "EFA_INSTALLER_VERSION")"
  if [[ -z "$efa_ver" ]]; then
    if grep -qE "aws-efa-installer-latest" "$file"; then
      finding ERROR "$file" "$(line_of "$file" 'aws-efa-installer-latest')" "EFA001" \
        "efa-installer pulled via 'latest' tarball -- builds are not reproducible." \
        "Pin ENV EFA_INSTALLER_VERSION=<x.y.z> and interpolate it into the URL."
    else
      finding WARN "$file" "$(line_of "$file" 'efa.installer|efa_installer')" "EFA001" \
        "No EFA_INSTALLER_VERSION pin found." \
        "CONTRIBUTING.md requires fixed versions, not 'latest', for reproducibility."
    fi
  else
    emit "  ${C_GRN}ok${C_RST}    EFA_INSTALLER_VERSION=${efa_ver}"

    # EFA002: the cheatsheet's documented NCCL memory-registration failure.
    if ver_lt "$efa_ver" "$EFA_CUDA_SYNC_MEMOPS_FLOOR"; then
      if ! grep -qE "FI_EFA_SET_CUDA_SYNC_MEMOPS" "$file"; then
        finding ERROR "$file" "$(line_of "$file" 'EFA_INSTALLER_VERSION')" "EFA002" \
          "efa-installer ${efa_ver} < ${EFA_CUDA_SYNC_MEMOPS_FLOOR} and FI_EFA_SET_CUDA_SYNC_MEMOPS is unset." \
          "With nccl>=2.19.0 this throws 'register_rail_mr_buffer ... Unable to register memory'. Set ENV FI_EFA_SET_CUDA_SYNC_MEMOPS=0 or upgrade efa-installer."
      fi
    fi
  fi

  # ---------- container-only installer flags ----------
  if grep -qE "efa_installer\.sh" "$file"; then
    local inst_line; inst_line="$(line_of "$file" 'efa_installer\.sh')"
    for flag in "${CONTAINER_ONLY_FLAGS[@]}"; do
      if ! grep -qE -- "$flag" "$file"; then
        finding ERROR "$file" "$inst_line" "EFA003" \
          "efa_installer.sh invoked without ${flag}." \
          "Inside a container the kmod build and verification step fail; the image build will error. Add ${flag} (container ONLY -- never on a host)."
      fi
    done
    if ! grep -qE -- "--skip-limit-conf" "$file"; then
      finding WARN "$file" "$inst_line" "EFA004" \
        "efa_installer.sh invoked without --skip-limit-conf." \
        "Redundant (the host already sets these limits) but conventional in this repo's container recipes."
    fi
  fi

  # ---------- aws-ofi-nccl ----------
  ofi_ver="$(extract_var "$file" "AWS_OFI_NCCL_VERSION")"
  if [[ -z "$ofi_ver" ]]; then
    if grep -qE "aws-ofi-nccl" "$file"; then
      finding WARN "$file" "$(line_of "$file" 'aws-ofi-nccl')" "NCCL001" \
        "aws-ofi-nccl referenced but no AWS_OFI_NCCL_VERSION pin found." \
        "Pin the tag so the plugin version is reproducible."
    fi
  else
    emit "  ${C_GRN}ok${C_RST}    AWS_OFI_NCCL_VERSION=${ofi_ver}"

    # NCCL002: the tag-naming trap. Upstream dropped "-aws" at v1.14.0, so a
    # copied-forward "-aws" suffix on a newer version 404s at clone time --
    # and a missing suffix on an older version does the same.
    local bare="${ofi_ver#v}"; bare="${bare%-aws}"
    local has_aws=0; [[ "$ofi_ver" == *-aws ]] && has_aws=1
    local vline; vline="$(line_of "$file" 'AWS_OFI_NCCL_VERSION')"

    if ver_lt "$bare" "$OFI_NCCL_SUFFIX_DROP"; then
      if [[ "$has_aws" -eq 0 ]]; then
        finding ERROR "$file" "$vline" "NCCL002" \
          "aws-ofi-nccl ${ofi_ver} is below v${OFI_NCCL_SUFFIX_DROP} but lacks the '-aws' suffix." \
          "Tags before v${OFI_NCCL_SUFFIX_DROP} are named like 'v${bare}-aws'. The git clone will fail with 'Remote branch not found'."
      fi
    else
      if [[ "$has_aws" -eq 1 ]]; then
        finding ERROR "$file" "$vline" "NCCL002" \
          "aws-ofi-nccl ${ofi_ver} is at/above v${OFI_NCCL_SUFFIX_DROP} but still carries the '-aws' suffix." \
          "Upstream dropped '-aws' at v${OFI_NCCL_SUFFIX_DROP}. Use 'v${bare}'. This is the most common copy-forward error."
      fi
    fi

    # NCCL004: plugin must be built against the EFA libfabric, not a system one.
    if grep -qE "aws-ofi-nccl" "$file" && grep -qE "configure" "$file"; then
      if ! grep -qE -- "--with-libfabric[= ]*/opt/amazon/efa" "$file"; then
        finding WARN "$file" "$(line_of "$file" 'configure')" "NCCL004" \
          "aws-ofi-nccl configure does not point --with-libfabric at /opt/amazon/efa." \
          "Without it the plugin may link a system libfabric and silently lose EFA support."
      fi
    fi
  fi

  # ---------- HPC-X shadowing ----------
  # nvcr PyTorch images bundle HPC-X with its own OMPI and nccl_rdma_sharp_plugin.
  # Left in place, they shadow the EFA stack and NCCL falls back to TCP.
  # Scope: ONLY the NVIDIA images that actually bundle HPC-X (pytorch, nemo,
  # and the *-training images). The DCGM and CUDA base images do not ship
  # /opt/hpcx, so flagging them would be a false positive.
  if body_has "$body" "^[[:space:]]*FROM[[:space:]]+[^[:space:]]*nvcr\.io/nvidia/(pytorch|nemo)" \
     && body_has "$body" "aws-ofi-nccl"; then
    if ! body_has "$body" "rm -rf .*(hpcx|/usr/local/mpi)"; then
      finding ERROR "$file" "$(line_of "$file" 'nvcr\.io/nvidia')" "NCCL003" \
        "NVIDIA base image detected but bundled HPC-X / OMPI is not removed." \
        "Remove /opt/hpcx/ompi, /usr/local/mpi and /opt/hpcx/nccl_rdma_sharp_plugin, then ldconfig. Otherwise NCCL silently uses the bundled plugin instead of EFA."
    else
      # ENV002: removing OMPI without clearing OPAL_PREFIX leaves a dangling
      # prefix that breaks mpirun with a confusing 'opal_init failed'.
      if ! grep -qE "^\s*ENV\s+OPAL_PREFIX=\s*$|^\s*ENV\s+OPAL_PREFIX=$" "$file"; then
        finding WARN "$file" "$(line_of "$file" 'rm -rf .*hpcx')" "ENV002" \
          "Bundled OMPI removed but OPAL_PREFIX is not cleared." \
          "Add 'ENV OPAL_PREFIX=' (empty) or mpirun may fail with opal_init errors."
      fi
    fi
  fi

  # ---------- library path ----------
  if grep -qE "efa_installer\.sh" "$file"; then
    if ! grep -qE "LD_LIBRARY_PATH.*(/opt/amazon/efa/lib)" "$file"; then
      finding ERROR "$file" "$(line_of "$file" 'efa_installer\.sh')" "ENV001" \
        "EFA installed but /opt/amazon/efa/lib is not on LD_LIBRARY_PATH." \
        "Add: ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:\$LD_LIBRARY_PATH -- otherwise libfabric resolves elsewhere and EFA is unused."
    fi
  fi

  # ---------- base image pinning ----------
  base_img="$(grep -oP '^\s*FROM\s+\K\S+' "$file" 2>/dev/null | head -1)"
  if [[ -n "$base_img" ]]; then
    if [[ "$base_img" == *":latest" ]] || [[ "$base_img" != *:* && "$base_img" != *"\$"* ]]; then
      finding WARN "$file" "$(line_of "$file" '^\s*FROM')" "PIN001" \
        "Base image '${base_img}' uses a floating tag." \
        "CONTRIBUTING.md: fix versions via a tag or commit ID; do not use 'latest'."
    fi
  fi

  # ---------- optional: how far behind upstream ----------
  if [[ "$CHECK_LATEST" -eq 1 && -n "$ofi_ver" && -n "$LATEST_OFI" ]]; then
    local bare2="${ofi_ver#v}"; bare2="${bare2%-aws}"
    local latest_bare="${LATEST_OFI#v}"
    if ver_lt "$bare2" "$latest_bare"; then
      finding INFO "$file" "$(line_of "$file" 'AWS_OFI_NCCL_VERSION')" "NCCL900" \
        "aws-ofi-nccl ${ofi_ver} is behind the latest release ${LATEST_OFI}." \
        "Informational only. Upgrade deliberately and re-run your NCCL tests; do not chase releases in CI."
    fi
  fi
}

# -----------------------------------------------------------------------------
# inventory mode
# -----------------------------------------------------------------------------
print_inventory() {
  local files=("$@")
  printf '%-58s %-14s %-16s\n' "FILE" "EFA_INSTALLER" "AWS_OFI_NCCL"
  printf '%-58s %-14s %-16s\n' \
    "$(printf '%.0s-' {1..58})" "$(printf '%.0s-' {1..14})" "$(printf '%.0s-' {1..16})"

  local efa_seen=() ofi_seen=()
  for f in "${files[@]}"; do
    grep -qiE "efa.installer|efa_installer|aws-ofi-nccl" "$f" 2>/dev/null || continue
    local e o disp
    e="$(extract_var "$f" "EFA_INSTALLER_VERSION")"; e="${e:---}"
    o="$(extract_var "$f" "AWS_OFI_NCCL_VERSION")";  o="${o:---}"
    disp="$f"; [[ ${#disp} -gt 58 ]] && disp="...${disp: -55}"
    printf '%-58s %-14s %-16s\n' "$disp" "$e" "$o"
    [[ "$e" != "--" ]] && efa_seen+=("$e")
    [[ "$o" != "--" ]] && ofi_seen+=("$o")
  done

  echo
  echo "Distinct EFA_INSTALLER_VERSION values:"
  printf '%s\n' "${efa_seen[@]:-}" | grep -v '^$' | sort -V | uniq -c | sed 's/^/  /'
  echo "Distinct AWS_OFI_NCCL_VERSION values:"
  printf '%s\n' "${ofi_seen[@]:-}" | grep -v '^$' | sort -V | uniq -c | sed 's/^/  /'
  echo
  echo "${C_DIM}Drift across many files is not automatically wrong -- test cases may"
  echo "legitimately pin different versions. It IS worth a look when the spread is"
  echo "wide, because it usually means files were copied forward and never revisited.${C_RST}"
}

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
CHECK_LATEST=0
INVENTORY=0
STRICT=0
JSON_OUT=0
PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-latest) CHECK_LATEST=1; shift ;;
    --inventory)    INVENTORY=1; shift ;;
    --strict)       STRICT=1; shift ;;
    --json)         JSON_OUT=1; shift ;;
    --no-color)     C_RED=""; C_YEL=""; C_GRN=""; C_DIM=""; C_RST=""; shift ;;
    -h|--help)      usage; exit 0 ;;
    -V|--version)   echo "$SCRIPT_NAME - $APP_VERSION"; exit 0 ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      echo "Run '$SCRIPT_NAME --help' for usage." >&2
      exit 1 ;;
    *) PATHS+=("$1"); shift ;;
  esac
done

if [[ ${#PATHS[@]} -eq 0 ]]; then
  echo "ERROR: no PATH given." >&2
  echo "Run '$SCRIPT_NAME --help' for usage." >&2
  exit 1
fi

if [[ "$JSON_OUT" -eq 1 ]] && ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: --json requires jq on PATH." >&2; exit 1
fi

# collect target files
FILES=()
for p in "${PATHS[@]}"; do
  if [[ -f "$p" ]]; then
    FILES+=("$p")
  elif [[ -d "$p" ]]; then
    while IFS= read -r f; do FILES+=("$f"); done < <(
      find "$p" \( -name 'Dockerfile' -o -name 'Dockerfile.*' -o -name '*.dockerfile' \) \
           -not -path '*/.git/*' -type f | sort
    )
  else
    echo "ERROR: no such file or directory: $p" >&2; exit 1
  fi
done

if [[ ${#FILES[@]} -eq 0 ]]; then
  log "No Dockerfiles found under: ${PATHS[*]}"
  exit 0
fi

# optional online version resolution
LATEST_OFI=""
if [[ "$CHECK_LATEST" -eq 1 ]]; then
  if ! command -v curl >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
    log "WARNING: --check-latest needs curl and jq; skipping online lookup."
    CHECK_LATEST=0
  else
    LATEST_OFI="$(curl -sf --max-time 10 \
      'https://api.github.com/repos/aws/aws-ofi-nccl/releases?per_page=10' \
      | jq -r '[.[] | select(.prerelease==false)][0].tag_name' 2>/dev/null || true)"
    if [[ -z "$LATEST_OFI" || "$LATEST_OFI" == "null" ]]; then
      log "WARNING: could not resolve latest aws-ofi-nccl release (offline or rate-limited)."
      CHECK_LATEST=0
    else
      log "Latest aws-ofi-nccl release: ${LATEST_OFI}"
    fi
  fi
fi

if [[ "$INVENTORY" -eq 1 ]]; then
  print_inventory "${FILES[@]}"
  exit 0
fi

[[ "$JSON_OUT" -eq 0 ]] && echo "Scanning ${#FILES[@]} Dockerfile(s)..." && echo

# NOTE: </dev/null is load-bearing. FILES may be populated by
# `while read ... < <(find ...)`, which leaves that process-substitution fd open.
# Checks below run `decomment "$file" | grep -q ...`, and grep can consume the
# leftover descriptor, causing a check to silently evaluate against the wrong
# input -- the same file then lints differently depending on whether it was
# passed directly or discovered via a directory scan. Detaching stdin makes
# every check deterministic.
for f in "${FILES[@]}"; do
  lint_file "$f" </dev/null
done

if [[ "$JSON_OUT" -eq 1 ]]; then
  jq -n --argjson findings "$FINDINGS_JSON" \
        --argjson scanned "${#FILES[@]}" \
        --argjson errors "$ERRORS" --argjson warnings "$WARNINGS" \
        --arg latest_ofi "${LATEST_OFI:-}" \
    '{scanned:$scanned, errors:$errors, warnings:$warnings,
      latest_aws_ofi_nccl:(if $latest_ofi=="" then null else $latest_ofi end),
      findings:$findings}'
else
  echo
  echo "Scanned ${#FILES[@]} file(s): ${C_RED}${ERRORS} error(s)${C_RST}, ${C_YEL}${WARNINGS} warning(s)${C_RST}"
  if [[ "$ERRORS" -eq 0 && "$WARNINGS" -eq 0 ]]; then
    echo "${C_GRN}No EFA/NCCL configuration issues detected.${C_RST}"
  fi
fi

if [[ "$ERRORS" -gt 0 ]]; then exit 1; fi
if [[ "$STRICT" -eq 1 && "$WARNINGS" -gt 0 ]]; then exit 1; fi
exit 0
