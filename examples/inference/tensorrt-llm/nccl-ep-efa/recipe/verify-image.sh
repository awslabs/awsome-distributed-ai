#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# Smoke the EFA + NcclEP substrate in the built image BEFORE loading a model. Fails loud.
# Static image check: asserts what the image stages (libs, symbols, imports, patch-marker
# consistency). The live cross-node transport proof is run-kernel-test.sh, not this.
set -euo pipefail
IMG="${1:?usage: verify-image.sh <image>}"

# EFA device mapping: fi_info -p efa only resolves with the device visible in the
# container. On an EFA host, pass /dev/infiniband through; elsewhere fall back to a
# provider-compiled-in check (fi_info -l) and say so.
DEV_ARGS=()
HAVE_EFA_DEV=0
if [ -d /dev/infiniband ]; then
  DEV_ARGS=(-v /dev/infiniband:/dev/infiniband --device=/dev/infiniband)
  HAVE_EFA_DEV=1
fi

docker run --rm --gpus all "${DEV_ARGS[@]}" -e HAVE_EFA_DEV="${HAVE_EFA_DEV}" "${IMG}" bash -lc '
  set -euo pipefail
  if [ "${HAVE_EFA_DEV}" = "1" ]; then
    echo "== fi_info efa (live fabric) =="
    [ "$(/opt/amazon/efa/bin/fi_info -p efa | grep -c "fabric: efa-direct")" -ge 1 ] || { echo "FAIL: no efa-direct"; exit 1; }
  else
    echo "== fi_info efa (no /dev/infiniband on this host — checking provider is compiled in only) =="
    /opt/amazon/efa/bin/fi_info -l | grep -qi "efa" || { echo "FAIL: efa provider not in libfabric"; exit 1; }
    echo "   (run this script on an EFA host for the live efa-direct fabric check)"
  fi
  echo "== single libnccl 2.30.4 wins (path + version string — the NGC base bakes an older NCCL in a DIFFERENT dir, so path matters) =="
  # draining form (awk NR==1, no `head`): `head -1` closes the pipe after one line, so grep
  # takes SIGPIPE-141 under pipefail — worst exactly when there are 2+ libnccl entries (the
  # baked-shadows-pip case this check exists to catch). Same trap the Dockerfile calls out.
  NCCL_SO=$(ldconfig -p | grep "libnccl.so.2 " | awk "NR==1{print \$NF}")
  echo "$NCCL_SO" | grep -q "nvidia/nccl" || { echo "FAIL: baked libnccl shadows pip ($NCCL_SO)"; exit 1; }
  [ "$(strings "$NCCL_SO" | grep -c "NCCL version 2.30.4")" -ge 1 ] || { echo "FAIL: $NCCL_SO is not 2.30.4 — NcclEP is_nccl_ep_installed() gates on >= 2.30.4"; exit 1; }
  echo "== GIN plugin symbol =="; [ "$(nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -c ncclGinPlugin)" -ge 1 ] || { echo "FAIL: no ncclGinPlugin"; exit 1; }
  echo "== nccl.ep python package (nccl4py) =="
  python3 -c "import nccl.ep; print(\"nccl.ep at\", nccl.ep.__file__)" || { echo "FAIL: import nccl.ep"; exit 1; }
  echo "== tensorrt_llm + the NcclEP factory module =="
  python3 -c "import tensorrt_llm; print(\"tensorrt_llm\", tensorrt_llm.__version__)"
  python3 -c "from tensorrt_llm._torch.modules.fused_moe.communication.communication_factory import CommunicationFactory; from tensorrt_llm._torch.modules.fused_moe.communication.nccl_ep import NcclEP; print(\"factory + NcclEP import OK\")" \
    || { echo "FAIL: NcclEP factory modules missing — wrong base tag? (GA v1.2.1 has no NcclEP backend)"; exit 1; }
  echo "== patch-marker consistency (marker and site-packages must agree) =="
  if [ -f /opt/.ht-flat-patch-applied ]; then
    SP=$(python3 -c "import tensorrt_llm, pathlib; print(pathlib.Path(tensorrt_llm.__file__).parent)")
    [ "$(grep -rc TRTLLM_NCCL_EP_ALGO "$SP/_torch/modules/fused_moe/communication/" 2>/dev/null | awk -F: "{s+=\$2} END {print s+0}")" -ge 1 ] \
      || { echo "FAIL: marker says patched but TRTLLM_NCCL_EP_ALGO gate not in site-packages"; exit 1; }
    echo "   patched image (PR#17715 baked — algorithm/layout selectable)"
  else
    echo "   unpatched baseline (upstream LOW_LATENCY+RANK_MAJOR — the measured-clean default)"
  fi
  echo "ALL CHECKS PASS"
'
