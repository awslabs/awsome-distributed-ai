<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Kimi2.6 — 1P1D on H200 (EKS / HyperPod)

Node-level prefill/decode disaggregation with SGLang on **two `ml.p5en.48xlarge`
(8×H200) nodes** — one prefill StatefulSet, one decode StatefulSet, KV cache
over **NIXL (EFA libfabric)**, fronted by the SGLang router. Optional DCGM +
Prometheus-agent → Amazon Managed Prometheus monitoring is included.

## Files

| File | Purpose |
|---|---|
| [`Dockerfile`](./Dockerfile) | `lmsysorg/sglang:dev-cu13` + AWS EFA installer |
| [`build-image.sh`](./build-image.sh) | build and push to ECR |
| [`kimi-pd-deploy.yaml`](./kimi-pd-deploy.yaml) | prefill + decode StatefulSets, router, AMP monitoring |

Model pre-staging and GPU metrics use the shared helpers one level up
([`../download-model.sh`](..), [`../dcgm-exporter-daemonset.yaml`](..)).

## Deploy

```bash
./build-image.sh                              # build + push, prints the image URI
# edit kimi-pd-deploy.yaml: set the image + (optional) AMP/IAM values below

../download-model.sh moonshotai/Kimi-K2.5 ml.p5en.48xlarge   # wait for "Download complete!"

kubectl apply -f kimi-pd-deploy.yaml
kubectl apply -f ../dcgm-exporter-daemonset.yaml   # optional monitoring
```

The router exposes an OpenAI-compatible endpoint on `sglang-router:30000`
(`ClusterIP`) — port-forward or front it with an ingress to call it:

```bash
kubectl port-forward svc/sglang-router 30000:30000
curl http://localhost:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "moonshotai-Kimi-K2.5", "prompt": "The capital of France is", "max_tokens": 32}'
```

## Replace before deploying

`kimi-pd-deploy.yaml` has `<PLACEHOLDER>` markers for everything
environment-specific — fill them in:

- **`<YOUR_ECR_IMAGE>`** (both StatefulSets) — the URI `build-image.sh` printed.
- **Model path** — `--model-path /nvme/moonshotai-Kimi-K2.5` and the
  `download-model-daemonset.yaml` `repo_id` (set to the real Kimi2.6 HF id).
- **Monitoring (optional)** — `<YOUR_AMP_INGEST_ROLE_ARN>`,
  `<YOUR_AMP_WORKSPACE_ID>`, and `<region>` in the Prometheus-agent block.
  Delete that block entirely if you don't use AMP.

Per-engine SGLang flags (`tp-size 8`, `ep-size 8`, `mem-fraction-static 0.92`,
hierarchical cache, parsers) live inline in the StatefulSet `command:` blocks.
