#!/usr/bin/env bash
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# selftest.sh — regression suite for efa-nccl-doctor.sh
#
# Run this after any change to the linter. It builds synthetic Dockerfiles in a
# temp dir, asserts the expected finding codes fire (and that clean files stay
# silent), and checks the exit-code contract that CI depends on.
#
#   ./selftest.sh
#
# Exit 0 = all assertions passed.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCTOR="${HERE}/efa-nccl-doctor.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0

ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }

# assert_code <fixture-name> <expected-code|NONE> <description>
assert_code() {
  local name="$1" expect="$2" desc="$3"
  local out; out="$("$DOCTOR" --no-color "$TMP/Dockerfile.$name" 2>&1)"
  if [[ "$expect" == "NONE" ]]; then
    if grep -qE "ERROR|WARN " <<<"$out"; then
      bad "$desc (expected clean, got findings)"
      sed 's/^/       /' <<<"$out" | grep -E "ERROR|WARN " || true
    else
      ok "$desc"
    fi
  else
    if grep -q "$expect" <<<"$out"; then ok "$desc"
    else
      bad "$desc (expected $expect)"
      sed 's/^/       /' <<<"$out" | grep -E "ERROR|WARN " || true
    fi
  fi
}

# assert_exit <expected> <description> <args...>
assert_exit() {
  local expect="$1" desc="$2"; shift 2
  "$DOCTOR" "$@" >/dev/null 2>&1
  local rc=$?
  [[ "$rc" -eq "$expect" ]] && ok "$desc (exit $rc)" || bad "$desc (exit $rc, expected $expect)"
}

# ---------- fixtures ----------
# A fully correct EFA container recipe. Every other fixture is this, with one
# thing broken, so a finding can only be attributable to that one change.
cat > "$TMP/Dockerfile.good" <<'EOF'
FROM nvcr.io/nvidia/pytorch:24.09-py3
ENV EFA_INSTALLER_VERSION=1.48.0
ENV AWS_OFI_NCCL_VERSION=v1.19.0
RUN rm -rf /opt/hpcx/ompi /usr/local/mpi /opt/hpcx/nccl_rdma_sharp_plugin && ldconfig
ENV OPAL_PREFIX=
RUN cd /tmp && ./efa_installer.sh -y -g -d --skip-kmod --no-verify --skip-limit-conf
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
RUN git clone -b ${AWS_OFI_NCCL_VERSION} https://github.com/aws/aws-ofi-nccl.git && ./configure --with-libfabric=/opt/amazon/efa
EOF

# NCCL002: pre-v1.14.0 tag missing the required -aws suffix (clone 404s).
sed 's|ENV AWS_OFI_NCCL_VERSION=v1.19.0|ENV AWS_OFI_NCCL_VERSION=v1.7.3|' \
  "$TMP/Dockerfile.good" > "$TMP/Dockerfile.suffix_missing"

# NCCL002: post-v1.14.0 tag carrying a stale -aws suffix (copy-forward error).
sed 's|ENV AWS_OFI_NCCL_VERSION=v1.19.0|ENV AWS_OFI_NCCL_VERSION=v1.19.0-aws|' \
  "$TMP/Dockerfile.good" > "$TMP/Dockerfile.suffix_stale"

# EFA002: efa-installer < 1.29.1 without FI_EFA_SET_CUDA_SYNC_MEMOPS=0.
sed 's|ENV EFA_INSTALLER_VERSION=1.48.0|ENV EFA_INSTALLER_VERSION=1.28.0|' \
  "$TMP/Dockerfile.good" > "$TMP/Dockerfile.memops"

# EFA003: container-only installer flag removed.
sed 's| --skip-kmod||' "$TMP/Dockerfile.good" > "$TMP/Dockerfile.no_skipkmod"

# NCCL003: bundled HPC-X left in place on an NVIDIA base.
grep -v 'rm -rf /opt/hpcx' "$TMP/Dockerfile.good" > "$TMP/Dockerfile.hpcx"

# ENV001: EFA installed but never put on LD_LIBRARY_PATH.
grep -v 'LD_LIBRARY_PATH' "$TMP/Dockerfile.good" > "$TMP/Dockerfile.no_ldpath"

# EFA001: floating 'latest' tarball instead of a pinned version.
sed 's|ENV EFA_INSTALLER_VERSION=1.48.0|RUN curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz|' \
  "$TMP/Dockerfile.good" > "$TMP/Dockerfile.latest_tarball"

# PIN001: floating base image tag.
sed 's|nvcr.io/nvidia/pytorch:24.09-py3|nvcr.io/nvidia/pytorch:latest|' \
  "$TMP/Dockerfile.good" > "$TMP/Dockerfile.floating_base"

# Out of scope: no EFA references at all — must be skipped, not flagged.
cat > "$TMP/Dockerfile.noefa" <<'EOF'
FROM python:3.12-slim
RUN pip install requests
EOF

# Comment-only mentions must NOT trigger checks. This is the false positive
# that a naive grep-based linter produces on real repo files.
# NOTE: the RUN line must NOT mention EFA, or the file is legitimately in
# scope. This fixture models the real cosmos3 Dockerfile, where the only
# mentions of nvcr.io/nvidia/pytorch and aws-ofi-nccl are in a comment block
# explaining why the AWS DLC base was chosen instead.
cat > "$TMP/Dockerfile.comment_only" <<'EOF'
# We deliberately avoid nvcr.io/nvidia/pytorch here because its bundled
# aws-ofi-nccl is built against an older NCCL and causes an ABI mismatch.
# The efa_installer.sh step is unnecessary: the DLC base ships EFA already.
FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.10.0-gpu-py313
RUN pip install --no-cache-dir transformers==4.57.0
EOF

# Suppression pragma must silence a genuine finding.
{ head -1 "$TMP/Dockerfile.hpcx"
  echo "# efa-nccl-doctor: disable=NCCL003 reason=torch links libmpi.so.40 from hpcx"
  tail -n +2 "$TMP/Dockerfile.hpcx"; } > "$TMP/Dockerfile.suppressed"

# ---------- checks ----------
echo "efa-nccl-doctor selftest"
echo
echo "detection:"
assert_code good            NONE      "clean recipe produces no findings"
assert_code suffix_missing  NCCL002   "pre-v1.14.0 tag without -aws suffix"
assert_code suffix_stale    NCCL002   "post-v1.14.0 tag with stale -aws suffix"
assert_code memops          EFA002    "old efa-installer without CUDA_SYNC_MEMOPS"
assert_code no_skipkmod     EFA003    "efa_installer.sh missing --skip-kmod"
assert_code hpcx            NCCL003   "bundled HPC-X not removed"
assert_code no_ldpath       ENV001    "EFA not on LD_LIBRARY_PATH"
assert_code latest_tarball  EFA001    "floating 'latest' EFA tarball"
assert_code floating_base   PIN001    "floating base image tag"

echo
echo "false-positive guards:"
assert_code noefa           NONE      "non-EFA Dockerfile is skipped"
assert_code comment_only    NONE      "comment-only mentions do not trigger checks"
assert_code suppressed      NONE      "disable= pragma silences a real finding"

echo
echo "exit-code contract:"
assert_exit 0 "clean file"            --no-color "$TMP/Dockerfile.good"
assert_exit 1 "file with an error"    --no-color "$TMP/Dockerfile.hpcx"
assert_exit 1 "missing path"          --no-color "$TMP/does-not-exist"
assert_exit 1 "no arguments"
assert_exit 0 "--help"                --help

echo
echo "invocation consistency:"
# The regression that motivated body_has(): a file passed directly and the same
# file found by a directory scan must produce identical findings. Process
# substitution used to leave an fd open that `grep -q` consumed.
mkdir -p "$TMP/scan" && cp "$TMP/Dockerfile.hpcx" "$TMP/scan/Dockerfile"
A="$("$DOCTOR" --no-color "$TMP/scan/Dockerfile" 2>&1 | grep -c 'NCCL003')"
B="$("$DOCTOR" --no-color "$TMP/scan"            2>&1 | grep -c 'NCCL003')"
if [[ "$A" == "$B" && "$A" == "1" ]]; then
  ok "file arg and directory scan agree (both found NCCL003)"
else
  bad "file arg ($A) and directory scan ($B) disagree"
fi

echo
echo "modes:"
"$DOCTOR" --inventory --no-color "$TMP" >/dev/null 2>&1 \
  && ok "--inventory runs" || bad "--inventory failed"
if command -v jq >/dev/null 2>&1; then
  if "$DOCTOR" --json "$TMP/Dockerfile.hpcx" 2>/dev/null | jq -e '.findings|length > 0' >/dev/null; then
    ok "--json emits parseable findings"
  else
    bad "--json output not parseable"
  fi
else
  echo "  SKIP --json (jq not installed)"
fi

echo
echo "-----------------------------------------"
printf 'passed: %d   failed: %d\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
