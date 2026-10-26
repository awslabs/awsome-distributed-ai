# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# $(( )) reads a leading zero as octal, so 010 silently becomes 8 and 08 fails outright; on other
# non-numeric input
# it either fails or evaluates the string as an expression. Both report from bash, naming neither
# the variable nor the fix, so check first.
for _v in COLOCATE ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS \
          ROLLOUT_GPUS_PER_ENGINE GPUS_PER_WORKER; do
  eval "_val=\${$_v:-}"
  case "${_v}" in
    COLOCATE)
      case "${_val}" in
        true|false) ;;
        *) echo "[ERROR] COLOCATE must be exactly 'true' or 'false', got '${_val}'." >&2
           unset _v _val; return 1 2>/dev/null || exit 1 ;;
      esac ;;
    *)
      case "${_val}" in
        ""|*[!0-9]*|0|0*)
          echo "[ERROR] ${_v} must be a positive integer with no leading zero, got '${_val}'." >&2
          unset _v _val; return 1 2>/dev/null || exit 1 ;;
      esac ;;
  esac
done
unset _v _val

# WORKER_REPLICAS: GPU worker nodes the RayCluster launches. Colocated shares one pool, so it is
# the actor node count. Disaggregated sizes the two pools separately and adds them, rather than
# dividing the GPU total: a bundle cannot span workers, so an actor of 1 x 7 GPU beside a rollout
# of 9 GPU in 3-GPU engines totals 16 and divides into two workers, yet after the actor lands only
# 1 + 8 GPUs are free and the third engine has nowhere to go.
if [ "${COLOCATE}" = "true" ]; then
  export WORKER_REPLICAS="${ACTOR_NUM_NODES}"
else
  if [ "$(( ROLLOUT_GPUS_PER_ENGINE > GPUS_PER_WORKER ))" = "1" ]; then
    echo "[ERROR] ROLLOUT_GPUS_PER_ENGINE=${ROLLOUT_GPUS_PER_ENGINE} exceeds" >&2
    echo "[ERROR] GPUS_PER_WORKER=${GPUS_PER_WORKER}; one engine cannot span workers." >&2
    return 1 2>/dev/null || exit 1
  fi
  _engines=$(( ( ROLLOUT_NUM_GPUS + ROLLOUT_GPUS_PER_ENGINE - 1 ) / ROLLOUT_GPUS_PER_ENGINE ))
  _per_worker=$(( GPUS_PER_WORKER / ROLLOUT_GPUS_PER_ENGINE ))
  export WORKER_REPLICAS=$(( ACTOR_NUM_NODES + ( _engines + _per_worker - 1 ) / _per_worker ))
  unset _engines _per_worker
fi

# CLUSTER_GPUS: what the RayCluster this env renders will actually have. The recipes refuse a
# larger request rather than letting Ray wait on a placement group it can never satisfy, which
# reads as a hang. Derived, because a ceiling nobody sets is a ceiling that never fires.
export CLUSTER_GPUS=$(( WORKER_REPLICAS * GPUS_PER_WORKER ))
