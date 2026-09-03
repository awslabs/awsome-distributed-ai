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

  echo "== the libnccl the LOADER resolves is the GIN build (path + version) =="
  # Ask the loader, not the ld.so cache. The image sets LD_LIBRARY_PATH to the
  # GIN build, and dlopen searches LD_LIBRARY_PATH BEFORE the cache, so
  # "ldconfig -p | head -1" (cache order) is not what decides resolution.
  # dlopen("libnccl.so.2") reproduces the real loader search (RPATH,
  # LD_LIBRARY_PATH, LD_PRELOAD, then cache); /proc/self/maps then reports the
  # file actually mapped — the same evidence deep_ep.check_nccl_so() acts on at
  # import. This catches a scrubbed LD_LIBRARY_PATH or an LD_PRELOAD shadow,
  # which the cache-order check reads right past.
  NCCL_SO=$(python3 -c "import ctypes, pathlib; ctypes.CDLL(\"libnccl.so.2\"); print(next(l.split()[-1] for l in pathlib.Path(\"/proc/self/maps\").read_text().splitlines() if \"libnccl.so.2\" in l))")
  echo "$NCCL_SO" | grep -q "/opt/nccl/build" || { echo "FAIL: loader resolves libnccl to $NCCL_SO, not the GIN build under /opt/nccl/build (LD_LIBRARY_PATH scrubbed, or a baked libnccl shadows it)"; exit 1; }
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
    # Probe one needle PER independently-required change, not one per PR: a
    # partially-merged tree satisfies a single needle while lacking later commits
    # (e.g. EP_EFA_MAX_QPS present but the get_rdma_gbs EFA fast path absent;
    # ElasticBuffer present but the num_experts backward-dispatch fix absent).
    # This mirrors the per-change probe list in patches/apply_nemo_rl_patches.py.
    DEEP_EP_DIR=$(python3 -c "import deep_ep, pathlib; print(pathlib.Path(deep_ep.__file__).parent)")
    grep -q EP_EFA_MAX_QPS "$DEEP_EP_DIR/buffers/elastic.py" \
      || { echo "FAIL: marker says patched but DeepEP#612 QP cap (EP_EFA_MAX_QPS) not in installed deep_ep"; exit 1; }
    grep -q EP_EFA_RDMA_GBS "$DEEP_EP_DIR/utils/envs.py" \
      || { echo "FAIL: marker says patched but DeepEP#612 get_rdma_gbs EFA fast path (EP_EFA_RDMA_GBS) not in installed deep_ep — partially-applied PR"; exit 1; }
    MEGA_A2A=/opt/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py
    grep -q ElasticBuffer "$MEGA_A2A" \
      || { echo "FAIL: marker says patched but Megatron-LM#4632 ElasticBuffer support not in the flex dispatcher"; exit 1; }
    grep -q _handle_num_experts "$MEGA_A2A" \
      || { echo "FAIL: marker says patched but Megatron-LM#4632 num_experts backward-dispatch fix absent — partially-applied PR"; exit 1; }
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
