#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
# Smoke the EFA + DeepEP-V2 + NeMo-RL substrate in the built image BEFORE any cluster
# deploy. Fails loud; there is no unconditional PASS in this file.
# Static image gate: asserts what the image stages (libs, symbols, imports, patch-marker
# consistency). The live cross-node transport proof is run-rollout-probe.sh, not this.
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

  echo "== single GIN-capable libnccl wins (path + version string — the NGC base bakes its own NCCL in a DIFFERENT dir, so path matters) =="
  NCCL_SO=$(ldconfig -p | grep "libnccl.so.2 " | head -1 | awk "{print \$NF}")
  echo "$NCCL_SO" | grep -q "/opt/nccl/build" || { echo "FAIL: baked libnccl shadows the GIN build ($NCCL_SO)"; exit 1; }
  [ "$(strings "$NCCL_SO" | grep -c "NCCL version 2.30.4")" -ge 1 ] || { echo "FAIL: $NCCL_SO is not 2.30.4 — wrong NCCL resolved"; exit 1; }

  echo "== aws-ofi-nccl GIN plugin =="
  [ "$(nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -c ncclGinPlugin)" -ge 1 ] || { echo "FAIL: no ncclGinPlugin symbol"; exit 1; }
  [ "$(strings /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -c "GDRCopy support not available at compile time")" -eq 0 ] \
    || { echo "FAIL: plugin built without gdrcopy — GIN init will fail at run time"; exit 1; }

  echo "== deep_ep V2 (NCCL backend) import + ElasticBuffer =="
  python3 -c "from deep_ep import ElasticBuffer; import deep_ep; print(\"deep_ep at\", deep_ep.__file__)" \
    || { echo "FAIL: deep_ep import / ElasticBuffer missing — not an EPv2 build?"; exit 1; }

  echo "== nemo_rl + megatron.core imports =="
  python3 -c "import nemo_rl.algorithms.grpo; import nemo_rl.models.megatron; print(\"nemo_rl OK\")" \
    || { echo "FAIL: nemo_rl imports (algorithms.grpo / models.megatron)"; exit 1; }
  python3 -c "import megatron.core; import megatron.core.transformer.moe.fused_a2a; print(\"megatron.core OK, tree:\", megatron.core.__file__)" \
    || { echo "FAIL: megatron.core imports — is /opt/Megatron-LM on PYTHONPATH?"; exit 1; }

  echo "== patch-marker consistency (marker and trees/site-packages must agree) =="
  if [ -f /opt/.draft-rollout-patches-applied ]; then
    DEEP_EP_DIR=$(python3 -c "import deep_ep, pathlib; print(pathlib.Path(deep_ep.__file__).parent)")
    grep -q EP_EFA_MAX_QPS "$DEEP_EP_DIR/buffers/elastic.py" \
      || { echo "FAIL: marker says patched but DeepEP#612 EFA cap not in installed deep_ep"; exit 1; }
    grep -q ElasticBuffer /opt/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py \
      || { echo "FAIL: marker says patched but Megatron-LM#4632 not in the flex dispatcher"; exit 1; }
    test -f /opt/NeMo-RL/examples/configs/recipes/llm/aws-efa-grpo-qwen3-30ba3b-2n8g-megatron.yaml \
      || { echo "FAIL: marker says patched but NeMo-RL#2410 EFA recipe config missing"; exit 1; }
    echo "   patched image (3 draft PRs baked — full GRPO rollout-over-DeepEP path staged)"
  else
    DEEP_EP_DIR=$(python3 -c "import deep_ep, pathlib; print(pathlib.Path(deep_ep.__file__).parent)")
    if grep -q EP_EFA_MAX_QPS "$DEEP_EP_DIR/buffers/elastic.py" 2>/dev/null; then
      echo "FAIL: no marker but the DeepEP#612 patch IS present — ambiguous patch state"; exit 1
    fi
    echo "   unpatched baseline (upstream-only trees — probe/train-step run with explicit SM/QP counts)"
  fi
  echo "ALL CHECKS PASS"
'
