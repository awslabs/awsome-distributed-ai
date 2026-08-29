#!/usr/bin/env bash
set -euo pipefail

: "${EP_ARM:?}"
: "${QUALIFIED_PARENT_REFERENCE:?}"
: "${MCORE_COMMIT:?}"
: "${MCORE_DEEPEP_V2_EAGER_COMM_PATCH_SHA256:?}"
: "${MCORE_FUSED_A2A_PARENT_SHA256:?}"
: "${MCORE_FUSED_A2A_RESULT_SHA256:?}"

repo=/opt/Megatron-Bridge/3rdparty/Megatron-LM
source_file=${repo}/megatron/core/transformer/moe/fused_a2a.py
patch=/opt/benchmark/patches/mcore-deepep-v2-eager-comm.patch

test "$(git -C "${repo}" rev-parse HEAD)" = "${MCORE_COMMIT}"
echo "${MCORE_FUSED_A2A_PARENT_SHA256}  ${source_file}" | sha256sum --check --strict
echo "${MCORE_DEEPEP_V2_EAGER_COMM_PATCH_SHA256}  ${patch}" | sha256sum --check --strict
git -C "${repo}" apply --check "${patch}"
git -C "${repo}" apply "${patch}"
echo "${MCORE_FUSED_A2A_RESULT_SHA256}  ${source_file}" | sha256sum --check --strict
python3 -m py_compile "${source_file}"

printf '%s  %s\n' "${MCORE_DEEPEP_V2_EAGER_COMM_PATCH_SHA256}" "${patch}" \
  > /opt/benchmark/mcore-deepep-v2-eager-comm.patch.sha256
python3 - <<'PY'
import json
import os
from pathlib import Path

patch_sha = os.environ["MCORE_DEEPEP_V2_EAGER_COMM_PATCH_SHA256"]
parent = os.environ["QUALIFIED_PARENT_REFERENCE"]
backend_path = Path("/opt/benchmark/backend.json")
backend = json.loads(backend_path.read_text())
backend["mcore_deepep_v2_eager_comm_patch_sha256"] = patch_sha
backend["qualified_parent_reference"] = parent
backend_path.write_text(json.dumps(backend, indent=2, sort_keys=True) + "\n")

common_path = Path("/opt/benchmark/common-build-manifest.json")
common = json.loads(common_path.read_text())
common["mcore_deepep_v2_eager_comm_patch_sha256"] = patch_sha
common_path.write_text(json.dumps(common, indent=2, sort_keys=True) + "\n")
PY
EP_ARM="${EP_ARM}" /opt/benchmark/verify-image.sh
