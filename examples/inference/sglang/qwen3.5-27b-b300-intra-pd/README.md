<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Qwen3.5-27B — Intra-node PD on B300 (EKS / HyperPod)

Prefill/decode disaggregation **within a single 8-GPU B300 node** using SGLang.
All engines run in one pod on one node: six prefill instances (GPU 0–5) and
two decode instances (GPU 6–7), each `tp=1, dp=1` on one GPU, split via
`--base-gpu-id`. KV cache moves prefill → decode over **NIXL**, staying
intra-node. A router sidecar shares the pod network namespace and reaches every
engine on `127.0.0.1`.

## Deploy

```bash
kubectl apply -f qwen-pd-deploy.yaml
kubectl rollout status deploy/qwen35-intra-pd
```

Targets a `p6-b300.48xlarge` node (`nodeAffinity` in the manifest matches both
the bare EKS `p6-b300.48xlarge` and the HyperPod `ml.p6-b300.48xlarge`
instance-type label).

The router exposes an OpenAI-compatible endpoint on `qwen35-router:30080`
(`ClusterIP`) — port-forward to call it:

```bash
kubectl port-forward svc/qwen35-router 30080:30080
curl http://localhost:30080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3.5-27B", "prompt": "The capital of France is", "max_tokens": 32}'
```

Tear down with `kubectl delete -f qwen-pd-deploy.yaml`.

## Notes

- Weights are read from the node's NVMe at `/opt/dlami/nvme/huggingface`
  (mounted into the container as `~/.cache/huggingface`). Optionally pre-stage
  them with the shared [`../download-model.sh`](..):
  `../download-model.sh Qwen/Qwen3.5-27B ml.p6-b300.48xlarge`.
- Stock image `lmsysorg/sglang:v0.5.12.post1-cu130` — no custom build needed;
  intra-node NIXL transfer doesn't cross EFA.
- Per-engine ports: prefill `30000-30005`, decode `30010-30011`, bootstrap
  `9000-9005` / `9010-9011`, router `30080`.
- All knobs (model, `mem-fraction-static`, `context-length`, GPU split, NIXL
  backend) live inline in [`qwen-pd-deploy.yaml`](./qwen-pd-deploy.yaml).
- Intra-node PD uses UCX; the EFA-detected warning is expected/benign
