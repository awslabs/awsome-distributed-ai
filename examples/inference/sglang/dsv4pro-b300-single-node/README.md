<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# DeepSeek V4 Pro — Unified on B300 (EKS / HyperPod)

Single-node, non-disaggregated SGLang serving of **DeepSeek V4 Pro** on one
B300 node. One engine spans all 8 GPUs (`tp=8, dp=8, --enable-dp-attention`,
MXFP4 MoE, EAGLE speculative decoding).

## Deploy

```bash
kubectl apply -f dsv4pro-deploy.yaml
kubectl rollout status deploy/dsv4pro-unified
```

Targets `ml.p6-b300.48xlarge` nodes (`nodeSelector` in the manifest).

OpenAI-compatible endpoint on `dsv4pro:30000` (`ClusterIP`) — port-forward to
call it:

```bash
kubectl port-forward svc/dsv4pro 30000:30000
curl http://localhost:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-ai/DeepSeek-V4-Pro", "prompt": "The capital of France is", "max_tokens": 32}'
```

Tear down with `kubectl delete -f dsv4pro-deploy.yaml`.

## Benchmark

```bash
kubectl exec deploy/dsv4pro-unified -- \
  python3 -m sglang.bench_serving --backend sglang \
    --dataset-name random --num-prompts 1000 \
    --random-input 2048 --random-output 256 \
    --request-rate inf --max-concurrency 25
```

Reference numbers (`random`, input 2048 / output 256, `--request-rate inf`):

| Concurrency | Req/s | Total tok/s | Output tok/s | Median TTFT | Median TPOT | Mean E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 25  | 2.56  | 2,953  | 329.6   | 396 ms  | 56 ms  | 9.7 s  |
| 50  | 4.28  | 4,946  | 552.1   | 407 ms  | 84 ms  | 11.6 s |
| 75  | 5.2   | 6,003  | 670.1   | 418 ms  | 105 ms | 14.3 s |
| 100 | 6.45  | 7,452  | 831.9   | 475 ms  | 119 ms | 15.3 s |
| 150 | 7.77  | 8,974  | 1,001.8 | 500 ms  | 141 ms | 18.9 s |
| 200 | 9.99  | 11,535 | 1,287.6 | 592 ms  | 158 ms | 19.5 s |
| 300 | 12.95 | 14,954 | 1,669.3 | 4.4 s   | 143 ms | 22.0 s |
| 500 | 14.16 | 16,347 | 1,824.7 | 16.8 s  | 135 ms | 30.5 s |

Throughput keeps climbing to ~16k tok/s around concurrency 500, but TTFT
degrades sharply past ~300 concurrent requests on a single node.

All model and tuning knobs (env vars + serve flags) live inline in
[`dsv4pro-deploy.yaml`](./dsv4pro-deploy.yaml). Weights load from the node's
NVMe at `/opt/dlami/nvme/huggingface` — optionally pre-stage them with the
shared [`../download-model.sh`](..):

```bash
../download-model.sh deepseek-ai/DeepSeek-V4-Pro ml.p6-b300.48xlarge
```
