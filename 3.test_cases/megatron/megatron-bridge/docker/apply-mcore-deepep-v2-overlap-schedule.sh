#!/usr/bin/env bash
set -euo pipefail

: "${EP_ARM:?}"
: "${QUALIFIED_PARENT_REFERENCE:?}"
: "${MCORE_COMMIT:?}"
: "${MCORE_DEEPEP_V2_OVERLAP_SCHEDULE_PATCH_SHA256:?}"
: "${MCORE_COMMON_UTILS_PARENT_SHA256:?}"
: "${MCORE_COMMON_UTILS_RESULT_SHA256:?}"
: "${MCORE_FINE_GRAINED_PARENT_SHA256:?}"
: "${MCORE_FINE_GRAINED_RESULT_SHA256:?}"

repo=/opt/Megatron-Bridge/3rdparty/Megatron-LM
common_utils=${repo}/megatron/core/models/common/utils.py
fine_grained=${repo}/megatron/core/models/gpt/fine_grained_callables.py
patch=/opt/benchmark/patches/mcore-deepep-v2-overlap-schedule.patch

test "$(git -C "${repo}" rev-parse HEAD)" = "${MCORE_COMMIT}"
echo "${MCORE_COMMON_UTILS_PARENT_SHA256}  ${common_utils}" | sha256sum --check --strict
echo "${MCORE_FINE_GRAINED_PARENT_SHA256}  ${fine_grained}" | sha256sum --check --strict
echo "${MCORE_DEEPEP_V2_OVERLAP_SCHEDULE_PATCH_SHA256}  ${patch}" | sha256sum --check --strict
git -C "${repo}" apply --check "${patch}"
git -C "${repo}" apply "${patch}"
echo "${MCORE_COMMON_UTILS_RESULT_SHA256}  ${common_utils}" | sha256sum --check --strict
echo "${MCORE_FINE_GRAINED_RESULT_SHA256}  ${fine_grained}" | sha256sum --check --strict
python3 -m py_compile "${common_utils}" "${fine_grained}"

printf '%s  %s\n' "${MCORE_DEEPEP_V2_OVERLAP_SCHEDULE_PATCH_SHA256}" "${patch}" \
  > /opt/benchmark/mcore-deepep-v2-overlap-schedule.patch.sha256
python3 - <<'PY'
import json
import os
from pathlib import Path

patch_sha = os.environ["MCORE_DEEPEP_V2_OVERLAP_SCHEDULE_PATCH_SHA256"]
parent = os.environ["QUALIFIED_PARENT_REFERENCE"]
source_hashes = {
    "megatron/core/models/common/utils.py": os.environ["MCORE_COMMON_UTILS_RESULT_SHA256"],
    "megatron/core/models/gpt/fine_grained_callables.py": os.environ[
        "MCORE_FINE_GRAINED_RESULT_SHA256"
    ],
}

backend_path = Path("/opt/benchmark/backend.json")
backend = json.loads(backend_path.read_text())
backend["mcore_deepep_v2_overlap_schedule_patch_sha256"] = patch_sha
backend["overlap_schedule_source_sha256"] = source_hashes
backend["overlap_schedule_qualified_parent_reference"] = parent
backend_path.write_text(json.dumps(backend, indent=2, sort_keys=True) + "\n")

common_path = Path("/opt/benchmark/common-build-manifest.json")
common = json.loads(common_path.read_text())
common["mcore_deepep_v2_overlap_schedule_patch_sha256"] = patch_sha
common["overlap_schedule_source_sha256"] = source_hashes
common_path.write_text(json.dumps(common, indent=2, sort_keys=True) + "\n")
PY
EP_ARM="${EP_ARM}" /opt/benchmark/verify-image.sh
