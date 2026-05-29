<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# DeepSeek-V3 Disaggregated Inference with vLLM, UCCL-EP, and NIXL on EKS

This sample shows how to serve **DeepSeek-V3-0324** (671B parameters, 256 experts,
top-8 routing, MLA) with disaggregated **prefill / decode** across multiple
`p5en.48xlarge` (8×H200) nodes on Amazon EKS or SageMaker HyperPod EKS, using:

- **vLLM 0.21.0** for OpenAI-compatible serving
- **UCCL-EP** (`deepep_high_throughput`, `deepep_low_latency`) for MoE
  expert-parallel all-to-all over EFA
- **NIXL** (NVIDIA Inference Xfer Library) over `LIBFABRIC` for KV-cache
  transfer between prefill and decode pods
- A simple Python proxy (`toy_proxy_server.py` shipped in vLLM) for routing

It benchmarks configs from a single 8-GPU unified node up to a 6-node
1-prefill / 5-decode disaggregated topology, plus a UCCL-EP vs. AGRS
(`allgather_reducescatter`) backend comparison.

## Indicative results

> **Caveat.** The numbers below come from single-seed runs at
> `--num-prompts 50`. They are smoke-benchmark numbers, not statistically
> tight measurements — re-run with the multi-seed loop in
> [`recipe/benchmark.sh`](recipe/benchmark.sh) and compute confidence
> intervals before quoting them anywhere downstream. Treat the rankings
> below as directionally informative, not as research findings.

| Config       | Nodes | GPUs | Burst RPS | Burst tok/s | P50 TPOT @1RPS | P50 TPOT @burst | tok/s/GPU |
| ------------ | :---: | :--: | --------: | ----------: | -------------: | --------------: | --------: |
| 1N unified   |   1   |   8  |     13.49 |       3,454 |        21.83ms |         44.00ms |    431.55 |
| 1P+1D       |   2   |  16  |     15.94 |       4,082 |        20.93ms |         40.91ms |    255.10 |
| 1P+3D       |   4   |  32  |     17.22 |       4,408 |        18.21ms |         33.76ms |    137.75 |
| **1P+4D**   |   5   |  40  | **19.18** |   **4,910** |        17.81ms |         32.51ms |    122.75 |
| 1P+5D       |   6   |  48  |     17.16 |       4,392 |        17.81ms |         30.34ms |     91.51 |
| 1P+3D AGRS  |   4   |  32  |     10.62 |       2,721 |        23.30ms |         48.20ms |     85.03 |

Workload: `--random-input-len 1024 --random-output-len 256 --num-prompts 50`
via `vllm bench serve`.

Directional observations:

1. **Disaggregation pays off mostly in latency, not throughput.** Unified
   has the best `tok/s/GPU`; disaggregation lowers burst TPOT roughly
   44ms → 30ms but only modestly increases peak RPS.
2. **The 1P:4D ratio appears to be the knee** — beyond that, the single
   prefill node looks like the bottleneck.
3. **UCCL-EP (`deepep_high_throughput` + `deepep_low_latency`) outperforms
   AGRS** for this MLA + 256-expert MoE model in our runs (≈30% TPOT
   improvement is consistent with public UCCL-EP benchmarks; the burst-RPS
   delta is from a single-seed run and warrants reproduction).

## Architecture

```
                     ┌──────────────────────┐
                     │  Client / benchmark  │
                     └──────────┬───────────┘
                                │ HTTP (OpenAI API)
                     ┌──────────▼───────────┐
                     │  toy_proxy_server.py │   (Pod, hostNetwork)
                     │  port 8000           │
                     └────┬───────────┬─────┘
                  /v1/* prefill  /v1/* decode
                          │           │
            ┌─────────────▼─┐       ┌─▼─────────────┐  ... ┌──────────────┐
            │ Prefill Pod   │       │ Decode Pod 0  │       │ Decode Pod N │
            │ TP=4, DP=2    │       │ TP=4, DP=2    │       │ TP=4, DP=2   │
            │ EP=8          │       │ EP=8          │       │ EP=8         │
            │ deepep_ht     │       │ deepep_ll     │       │ deepep_ll    │
            │ port 8100     │       │ port 8200     │       │ port 8200    │
            └──────┬────────┘       └──────┬────────┘       └──────┬───────┘
                   │                       │                       │
                   └─────── NIXL KV cache (LIBFABRIC over EFA) ────┘
                            UCCL-EP all-to-all (EFA, intra-node NVLink)
```

- **Prefill** runs UCCL-EP `deepep_high_throughput` (high token/s under bursts)
- **Decode** runs UCCL-EP `deepep_low_latency` (low TPOT, CUDA graphs)
- **NIXL** transfers KV blocks from prefill → decode over EFA via
  `LIBFABRIC` backend
- All pods run with `hostNetwork: true` so EFA / NIXL / NCCL can use the
  full 16-EFA, 3,200 Gbps fabric

## Prerequisites

### Cluster

- Amazon EKS or SageMaker HyperPod EKS, version **1.33+** (1.32 and below
  are out of EKS standard support; 1.33 added native EFA traffic in the
  default EKS security group)
- 1–6 × `p5en.48xlarge` GPU nodes (8×H200 141GB, 16×EFA, ~30 TB NVMe)
- NVIDIA device plugin and EFA device plugin installed
- (Recommended) FSx for Lustre PVC for the model cache, or local NVMe under
  `/mnt/k8s-disks/0`

### Software

| Component | Version |
| --- | --- |
| Kubernetes | 1.33+ (EKS / HyperPod EKS tested) |
| kubectl | 1.33+ |
| Docker / BuildKit | 24.0+ |
| AWS CLI v2 | latest |
| vLLM | 0.21.0 |
| UCCL | commit `0dc87eb` |
| NIXL | v1.1.0 |
| PyTorch | 2.11.0 (cu130) |
| Ray | 2.55.1 |
| EFA installer | 1.48.0 |
| NCCL | v2.30.4-1 |

### Accounts and tokens

- Hugging Face token with access to `deepseek-ai/DeepSeek-V3-0324`
- Container registry (e.g. ECR) writable from your build host
- `kubectl` configured for your cluster
- Around ~680 GB of fast storage for the FP8 weights cache

## Repository layout

```
dsv3-uccl-nixl/
├── Dockerfile                  # vLLM 0.21.0 + NIXL 1.1.0 + UCCL-EP, CUDA 13
├── README.md                   # this file
├── setup/
│   ├── env_vars.example        # cluster + image + role config
│   ├── build-push.sh           # docker build + ECR push
│   └── install-prereqs.sh      # NVIDIA + EFA device plugins (optional)
├── manifests/
│   ├── prefill.yaml            # prefill Deployment (deepep_high_throughput)
│   ├── decode.yaml             # decode Deployment template (deepep_low_latency)
│   ├── proxy.yaml              # router Deployment (toy_proxy_server.py)
│   └── unified.yaml            # 1-node baseline Deployment (no NIXL)
└── recipe/
    ├── deploy.sh               # render + apply manifests for a chosen topology
    ├── teardown.sh             # delete Deployments
    └── benchmark.sh            # vllm bench serve rate sweep + warmup, multi-seed
```

## Quick start

### 1. Configure environment

```bash
cd 3.test_cases/pytorch/vllm/dsv3-uccl-nixl
cp setup/env_vars.example setup/env_vars
$EDITOR setup/env_vars              # replace every REPLACE_ME placeholder
grep REPLACE_ME setup/env_vars      # should print nothing when fully filled
source setup/env_vars
```

> The `env_vars` file holds your HuggingFace token and AWS account ID.
> It is gitignored — never commit it.

### 2. Build and push the image

```bash
./setup/build-push.sh
```

The image bundles:

- vLLM 0.21.0 (built from source for `sm_80 sm_86 sm_89 sm_90 sm_100 sm_103`)
- UCCL-EP with the `deep_ep` drop-in wrapper (so vLLM auto-detects it)
- NIXL 1.1.0 (CUDA-13 backend auto-selected from the meta wheel) with the
  `LIBFABRIC` backend
- Ray 2.55.1 (DP coordinator)
- EFA installer 1.48.0, NCCL 2.30.4, AWS OFI NCCL plugin
- GDRCopy 2.5.2

The build also runs smoke tests for `vllm`, `nixl`, `uccl.ep`, and `deep_ep`.

Expected size: ~22 GB. Build time: ~45–60 minutes on a 32-core host with a GPU.

### 3. Create namespace and HF secret

```bash
kubectl create namespace "$NAMESPACE"
kubectl create secret generic hf-token \
  --namespace "$NAMESPACE" \
  --from-literal=HF_TOKEN="$HF_TOKEN"
```

### 4. Discover node IPs

```bash
kubectl get nodes -l node.kubernetes.io/instance-type="$INSTANCE_TYPE" \
  -o custom-columns="NAME:.metadata.name,IP:.status.addresses[?(@.type=='InternalIP')].address" \
  --no-headers
```

Set the `PREFILL_NODE`, `DECODE_NODE_*`, and matching `*_IP` variables in
`setup/env_vars`, then re-`source` it.

### 5. Deploy a topology

`recipe/deploy.sh` accepts a topology name and renders manifests with the IPs
from `env_vars`:

```bash
# 1-node unified baseline (no NIXL, deepep_low_latency only)
./recipe/deploy.sh unified

# 1 prefill + 1 decode (2 nodes)
./recipe/deploy.sh 1p1d

# 1 prefill + 3 decode (4 nodes)
./recipe/deploy.sh 1p3d

# 1 prefill + 4 decode (5 nodes — peak burst RPS)
./recipe/deploy.sh 1p4d
```

Wait until the proxy logs `All instances ready!`. First deploy downloads
~680 GB of weights — allow 30–60 minutes.

### 6. Smoke test

```bash
curl -s "http://${PREFILL_IP}:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V3-0324",
    "prompt": "The capital of France is",
    "max_tokens": 32,
    "temperature": 0
  }' | python3 -m json.tool | head
```

### 7. Benchmark

`recipe/benchmark.sh` runs a warm-up then a rate sweep at 1, 4, 8, 16, and
`inf` RPS using `vllm bench serve`, looping over three seeds (1234, 5678,
9012) per rate so you can aggregate over multiple runs:

```bash
./recipe/benchmark.sh 1p4d
```

Results land in `results/<topology>/seed-<seed>/openai-<rate>qps-*.json`.
Key fields per JSON: `request_throughput`, `output_throughput`,
`p50_tpot_ms`, `p50_ttft_ms`. Aggregate across the three seed dirs and
report a confidence interval before quoting any number externally.

Override the seeds, rates, or prompt count with environment variables:

```bash
SEEDS="1234" RATES="1 inf" NUM_PROMPTS=200 ./recipe/benchmark.sh 1p4d
```

### 8. Tear down

```bash
./recipe/teardown.sh
```

## Configuration knobs

### Per-node parallelism (TP=4, DP=2, EP=8)

This is the per-pod parallelism setting we measured to be optimal for
DeepSeek-V3 on H200. It splits 8 GPUs into:

- 2 data-parallel replicas (`--data-parallel-size 2`)
- Each replica spans 4 GPUs with tensor parallelism (`--tensor-parallel-size 4`)
- All 8 GPUs participate in expert-parallel all-to-all (`--enable-expert-parallel`,
  EP=DP×TP=8)

You should not need to tune this for `p5en.48xlarge`. For other hardware
(H100 80GB, A100 80GB, B300) revisit `--gpu-memory-utilization`,
`--num-gpu-blocks-override`, and `--max-model-len`.

### All-to-all backend

The `--all2all-backend` flag chooses how MoE token-routing all-to-all is
implemented. For DeepSeek-V3 (MLA + 256 experts):

| Backend | When to use | Effect |
| --- | --- | --- |
| `deepep_high_throughput` | **Prefill nodes only** | High token/s, no CUDA graphs |
| `deepep_low_latency` | Decode and unified | Low TPOT, CUDA-graph captured |
| `allgather_reducescatter` (AGRS) | GQA models or fallback | 60–90% slower on MLA |

For DeepSeek-V3 always use `deepep_high_throughput` on prefill and
`deepep_low_latency` on decode — that is the configuration baked into
`manifests/prefill.yaml` and `manifests/decode.yaml`.

### NIXL KV transfer

```yaml
--kv-transfer-config '{"kv_connector":"NixlConnector",
                       "kv_role":"kv_both",
                       "kv_load_failure_policy":"fail",
                       "kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'
```

In vLLM 0.21.0, `kv_role` is a required-but-ignored placeholder for
`NixlConnector` — the [vLLM NixlConnector docs](https://docs.vllm.ai/en/v0.21.0/features/nixl_connector_usage.html)
state that "the actual prefiller/decoder roles are determined by the
upper-level proxy. Therefore, `kv_role` in `--kv-transfer-config` is
effectively a placeholder and does not affect NixlConnector's behavior."
What actually drives prefill vs. decode behavior is the proxy injecting
`do_remote_prefill` / `do_remote_decode` into per-request `kv_transfer_params`.
The config validator just refuses to start without *some* `kv_role`; we set
`kv_both` for that reason only.

`LIBFABRIC` uses EFA via the libfabric provider; alternatives like `UCX`
are not yet recommended on EFA without GPUDirect RDMA tuning.

`VLLM_NIXL_SIDE_CHANNEL_PORT=5600` and `VLLM_NIXL_SIDE_CHANNEL_HOST=$MY_IP`
expose the metadata exchange channel on the host network.

### Proxy

The proxy uses
`/opt/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`
which ships in the vLLM source tree. The vLLM NixlConnector docs direct
users to invoke it from that path, but it lives under `tests/` and is named
`toy_*` — upstream is free to move or delete it on any release. The image
pins `VLLM_COMMIT=v0.21.0`, so the path is stable for this sample, but
verify it before bumping that ARG.

It does simple round-robin over `--decoder-hosts` and routes the prefill
phase to `--prefiller-hosts`. For production traffic shaping you will want
a smarter router — this proxy exists to demonstrate the wire protocol, not
to be a load balancer.

### Storage

By default the manifests mount `/mnt/k8s-disks/0` (instance NVMe on
`p5en.48xlarge`) at `HF_HOME=/mnt/k8s-disks/0/local_scratch/hf_cache`. To
use FSx for Lustre instead, replace the `local-storage` volume with a PVC:

```yaml
volumes:
  - name: model-cache
    persistentVolumeClaim:
      claimName: fsx-claim
```

and set `HF_HOME` accordingly. Note: ~680 GB FP8 weights load in
~5 minutes from local NVMe; FSx adds ~1–2 minutes on the first node.

## Variables you must set

`setup/env_vars` controls every replaceable value. Required keys:

| Variable | Example | Purpose |
| --- | --- | --- |
| `AWS_REGION` | `us-west-2` | ECR + EC2 region |
| `ACCOUNT` | `123456789012` | AWS account ID for ECR |
| `REGISTRY` | `${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/` | ECR registry |
| `IMAGE` | `vllm-uccl-ep` | ECR repo name |
| `TAG` | `vllm0.21.0-uccl-0dc87eb-cu13` | image tag |
| `NAMESPACE` | `dsv3-disagg` | Kubernetes namespace |
| `INSTANCE_TYPE` | `p5en.48xlarge` | node selector value |
| `MODEL` | `deepseek-ai/DeepSeek-V3-0324` | HF model id |
| `HF_TOKEN` | `hf_…` | gated model access |
| `PREFILL_NODE` | `ip-10-…compute.internal` | hostname for prefill pod |
| `PREFILL_IP` | `10.…` | internal IP for proxy / NIXL |
| `DECODE_NODE_0..N` | hostname strings | one per decode pod |
| `DECODE_0_IP..N_IP` | IP strings | matching internal IPs |

## Required smoke tests before benchmarking

Inside any pod (`kubectl exec`):

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.device_count())"
python3 -c "import vllm; print(vllm.__version__)"
python3 -c "import uccl.ep"
python3 -c "import nixl"
nvidia-smi
fi_info -p efa
```

If `fi_info -p efa` returns no providers, `hostNetwork: true`, the EFA
device plugin, or the `vpc.amazonaws.com/efa: 16` resource limit are
misconfigured — fix that before benchmarking, otherwise NCCL silently
falls back to TCP and you get 28× slower throughput.

## Troubleshooting

### `NCCL timeout` during model load

Make sure `hostNetwork: true` is set on every pod and that
`NCCL_TIMEOUT=7200000` and `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200` are
exported (they are in the manifest). 680 GB takes a while to shard across
DP replicas the first time.

### `uccl_ep.cc:258 'out of memory'`

NVSHMEM symmetric buffer pre-allocation exceeded GPU memory. Lower
`--gpu-memory-utilization` from 0.80 to 0.78, or switch the backend to
`allgather_reducescatter` (slower but no symmetric heap).

### `NIXL: Failed to connect to peer`

- Verify `VLLM_NIXL_SIDE_CHANNEL_HOST` matches the pod's primary IP
- Verify the port `5600` is not bound on the host already
- Confirm `hostNetwork: true`

### Proxy stuck on "still waiting for …"

```bash
kubectl get pods -n "$NAMESPACE" -o wide
kubectl logs <pod-name> -n "$NAMESPACE" --tail=80
```

CUDA-graph capture can take 2–5 minutes after `/health` returns 200. Be
patient on the first benchmark.

## Cost

GPU instance pricing changes too often (and varies by region, term, and
Capacity Block) for a baked-in table to age well. Check the live source:

- [On-Demand pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
- [Capacity Block pricing](https://aws.amazon.com/ec2/capacityblocks/pricing/)
- Or query the AWS pricing API:
  ```bash
  aws pricing get-products --service-code AmazonEC2 \
    --filters Type=TERM_MATCH,Field=instanceType,Value=p5en.48xlarge \
              Type=TERM_MATCH,Field=location,Value="US West (Oregon)"
  ```

As a rough sense of scale: the largest topology in this sample uses 6
nodes, and a full rate sweep across topologies takes 4–6 hours end to end
(including the first-time ~680 GB model download).

## License

This sample is released under the [MIT-0 License](../../../../LICENSE), the
same license as the rest of `awsome-distributed-ai`.

## References

- vLLM NixlConnector usage:
  https://docs.vllm.ai/en/v0.21.0/features/nixl_connector_usage.html
- vLLM disaggregated serving:
  https://github.com/vllm-project/vllm/blob/main/docs/source/serving/disaggregated_serving.md
- UCCL-EP: https://github.com/uccl-project/uccl
- NIXL: https://github.com/ai-dynamo/nixl
- NIXL libfabric (EFA) plugin:
  https://github.com/ai-dynamo/nixl/blob/main/src/plugins/libfabric/README.md
- DeepSeek-V3 paper: https://arxiv.org/abs/2412.19437
- AWS EFA cheat sheet:
  https://github.com/awslabs/awsome-distributed-ai/blob/main/architectures/efa-cheatsheet.md
