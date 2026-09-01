<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# DeepSeek V4 Pro — Unified on B300 (HyperPod-on-EKS or self-managed EKS)

Single-node, non-disaggregated SGLang serving of **DeepSeek V4 Pro** on one
B300 node. One engine spans all 8 GPUs (`tp=8, dp=8, --enable-dp-attention`,
MXFP4 MoE, EAGLE speculative decoding).

## Prerequisite — provision the B300 node

On a **self-managed (eksctl) EKS** cluster, create the B300 spot nodegroup first
(labels, taint, and NVMe RAID0 auto-mount) — see
[**NODEGROUP-EKS.md**](./NODEGROUP-EKS.md) and
[`manifests/nodegroup-b300-eks.yaml`](./manifests/nodegroup-b300-eks.yaml). On
**HyperPod-on-EKS**, nodes are already provisioned — skip straight to Deploy.

## Deploy

A single manifest — [`manifests/dsv4pro-deploy.yaml`](./manifests/dsv4pro-deploy.yaml)
— runs on **both** HyperPod-on-EKS and self-managed EKS. Scheduling is handled
for both automatically: `nodeAffinity` matches the `p6-b300.48xlarge` instance
type with or without the HyperPod `ml.` prefix, and the pod always tolerates the
`nvidia.com/gpu=true:NoSchedule` taint the eksctl nodegroup applies (a no-op on
HyperPod nodes, which don't carry it).

The **one** thing you must set per environment is the NVMe `hostPath` near the
bottom of the manifest — the local disk is mounted at a different path on each
AMI. The default is HyperPod's; for self-managed EKS edit both `hostPath.path`
lines (the EKS value is shown inline as a comment).

| Cluster | NVMe `hostPath` |
|---|---|
| SageMaker **HyperPod-on-EKS** | `/opt/dlami/nvme/...` (DLAMI default — no edit needed) |
| Self-managed (**eksctl**) EKS | `/mnt/k8s-disks/0/...` (`setup-local-disks raid0`) |

> ⚠️ Picking the wrong path is not a hard error: `type: DirectoryOrCreate`
> silently creates an empty dir on the small root disk, and the 500GB+ weights
> then fill it up.

```bash
# On self-managed EKS, first complete the Prerequisite above, then edit the two
# NVMe hostPath lines in the manifest to /mnt/k8s-disks/0. Then:
kubectl apply -f manifests/dsv4pro-deploy.yaml

kubectl rollout status deploy/dsv4pro-unified
```

OpenAI-compatible endpoint on `dsv4pro:30000` (`ClusterIP`) — port-forward to
call it:

```bash
kubectl port-forward svc/dsv4pro 30000:30000
curl http://localhost:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-ai/DeepSeek-V4-Pro", "prompt": "The capital of France is", "max_tokens": 32}'
```

Tear down with `kubectl delete -f manifests/dsv4pro-deploy.yaml`.

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

All model and tuning knobs (env vars + serve flags) live inline in the manifest.
Weights are downloaded to the node's NVMe on first start
(`/opt/dlami/nvme/huggingface` on HyperPod, `/mnt/k8s-disks/0/huggingface` on
EKS — whichever `hostPath` you set above).
