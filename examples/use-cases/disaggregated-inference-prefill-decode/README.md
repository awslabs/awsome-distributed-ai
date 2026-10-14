# Disaggregated inference: when to split prefill from decode

This Amazon EKS lab compares two unified SGLang replicas behind a round-robin router with a prefill worker and a decode worker. Paired mode keeps both stacks resident in separate namespaces on disjoint GPU allocations. Each stack can pack its engine processes onto one node or use separate nodes for its workers. Both configurations use the same model, cache policy, GPU count, and latency objectives. Participants measure the cost of handing off the KV cache, then vary traffic to find where isolating decode from prefill helps useful throughput.

Prefill processes the input tokens in parallel and is often compute-bound. Decode repeatedly reads model weights and KV state to generate the next token and is often memory-bandwidth-bound. In a unified engine, a long prefill forward pass occupies the same GPUs as active decode steps; queued prefills can also raise TTFT. Chunked prefill reduces each interruption, while separate pools remove that source of shared-GPU interference. Neither arrangement removes queueing when its own capacity is exhausted.

Prepared for re:Invent 2026 session AIM345, led by Keita Watanabe with Mijanur Palash. The scripts are review material. Hardware results and remaining validation gaps are listed below; the expected ranking is a hypothesis for this configuration, not a benchmark transcript.

## Validation status

`VALIDATED` identifies an observed check within the stated scope. `UNVALIDATED` identifies an expected result or a step that still needs evidence. CPU checks do not establish GPU serving correctness.

| Check or expected result | Status | Scope |
|---|---|---|
| Image builds; installed SGLang and both NIXL distributions match the pins | VALIDATED | Local container build and package inspection; no GPU measurement |
| Agentic task chaining, token lengths, prefix isolation, streaming parsing, and SLO failure accounting | VALIDATED | CPU tests with a local mock stream; no inference performance measurement |
| CUDA-buffer transfer over NIXL LIBFABRIC/EFA with destination verification and matching RDMA byte counters | VALIDATED | Two Seoul `p6-b300.48xlarge` nodes, one GPU and one EFA device per peer; see [hardware evidence](VALIDATION.md) |
| Both selector defaults and absence of an explicit eager connection call in the pinned connector | VALIDATED | Inspection of installed package source; not a hardware negative-control experiment |
| Disaggregated deployment serves both traffic shapes | VALIDATED | Seoul `p6-b300.48xlarge`, short smoke runs at a GPU count of two; full workload/rate calibration remains UNVALIDATED |
| Matched unified deployment serves both traffic shapes | VALIDATED | Same Seoul model and GPU budget as the disaggregated smoke runs; two replicas behind explicit round-robin routing |
| Positive SGLang NIXL/LIBFABRIC request transfers KV over EFA | VALIDATED | Seoul `p6-b300.48xlarge`, one GPU per engine; streamed completion and EFA RDMA-write byte increase, supported by the separate GPU-buffer probe |
| Paired unified and disaggregated endpoints remain resident while both traffic shapes run | VALIDATED | Oregon EKS on 2026-09-09, two `g7e.12xlarge` nodes, two GPUs and one EFA device per stack, packed placement; cross-node KV/EFA transfer is outside this observation |
| Round 1: cached agentic traffic has worse TTFT after disaggregation | UNVALIDATED | Expected from prior work; not measured with this lab model and instance configuration |
| Round 1: measured wire time explains only a small part of the handoff penalty | UNVALIDATED | Requires measured fabric bandwidth and full-model KV accounting |
| Round 1: explicit `makeConnection` during SGLang bootstrap reduces the penalty | UNVALIDATED | Blocked on upstream SGLang support in the pinned release; no local patch |
| Round 2: unified violates the objective while disaggregated remains within it | UNVALIDATED | Offered rates and objectives require calibration on the approved production hardware |
| Add a prefill worker while preserving decode, then serve requests | VALIDATED | Seoul `p6-b300.48xlarge`, total GPU count increased from two to three; decode pod identity and restart count preserved |
| Round 2: adding prefill capacity alone restores the objective | UNVALIDATED | Requires an additional allocated worker and a prefill-bound operating point |
| Round 3: prompt length, offered rate, and measured cache hits identify a crossover | UNVALIDATED | Requires repeated paired measurements |

The production allocation is `g7.48xlarge` primary or `g7.24xlarge` secondary. DeepSeek-V2-Lite-Chat is the common model on both, with the original main-line engine pin. The separate optional V4 profile is restricted to the full primary allocation. Historical Seoul and Oregon rows retain their measured hardware labels.

## Prerequisites and pins

Use an existing EKS cluster with separately allocated, same-AZ GPU nodes, EFA interfaces attached at launch, compatible NVIDIA drivers, and working GPU and EFA device plugins. Inspect the actual GPU-memory registration and transfer evidence; EFA counter movement alone does not prove a path without host staging. Security groups must allow the EFA self-traffic required by the cluster architecture and TCP communication among the lab nodes for serving and bootstrap. The scripts create serving resources, a dedicated namespace, and its nonpreempting PriorityClass; they do not provision nodes or change cluster networking. Cluster administrators can use the repository's [EKS architecture](../../../architectures/sagemaker-hyperpod-eks/) as a platform reference.

The sequential path uses two nodes, with one worker per node. Configure GPU and EFA resource counts from the allocated nodes; the example requires those counts to be supplied. Packed paired mode puts both workers in one pod and requests twice `gpus_per_worker` GPUs and `efa_per_worker` EFA devices for that pod. The observed `g7e.12xlarge` allocation used one GPU per engine, two GPUs per stack and one EFA device per stack. Those counts do not describe `g7e.24xlarge`. Prefill scaling uses the separate-node path and adds an allocated worker.

The checked controller tool versions are Python version `3.12.3`, Docker version `29.7.2`, AWS CLI version `2.36.7`, and `kubectl` version `1.31.0`. Use a `kubectl` version compatible with your EKS API and record any version change. Python client dependencies, including transitive packages, are pinned in [requirements.txt](requirements.txt). Provision these tools before the timed lab. Model download, image pull, and kernel compilation can consume minutes; complete them before attendees arrive.

| Dependency | Pin | Purpose |
|---|---|---|
| SGLang CUDA image | `lmsysorg/sglang:v0.5.12.post1-cu130@sha256:ceaf8b16e02d165143633ac228bbb994a05fe77d7e0526cf035ae4bbf4eacc36` | Fixes inherited CUDA and serving dependencies |
| SGLang router | Version `0.3.2` with dependencies in `router-requirements.txt`, on the pinned Python image in `Dockerfile` | Separate CPU image avoids pulling CUDA onto small system-node disks |
| NIXL Python stub and CUDA distribution | `nixl==1.1.0`, `nixl-cu13==1.1.0` | Explicit reinstall and build/runtime assertions |
| EFA userspace installer | Version `1.47.0` | LIBFABRIC EFA provider; kernel modules stay on the host |
| Example MLA model | `deepseek-ai/DeepSeek-V2-Lite-Chat`, revision `85864749cd611b4353ce1decdb286193298f64c7` | Common main-line model on both G7 shapes |
| Traffic data | [shape-a.json](shape-a.json), [shape-b.json](shape-b.json) | Synthetic incident-analysis tasks authored with the lab; no external dataset download |

NIXL version `1.1.0` is deliberate. Prior field work reported a KV-transfer slowdown with version `1.2.0` over EFA, ending in `Decode transfer failed` and `KVPoll.WaitingForInput` timeouts. The upstream root cause remains unconfirmed; this pin is a conservative deployment choice, not an upstream-documented fix. The SGLang Dockerfile installs NIXL without a version pin, so a SGLang image tag alone is insufficient evidence of its installed NIXL version. `verify_image.py` inspects installed distributions during the build and before each engine starts. Do not replace this check with an inference from the image's build date.

## Prepare the image and configuration

Run commands from this directory. The registry below is an operator-supplied private ECR repository. Build and publish the image through your normal registry process, then use its digest in `config.json`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export LAB_IMAGE="${LAB_REGISTRY}/aim345:sglang-0.5.12.post1-nixl-1.1.0"
export LAB_ROUTER_IMAGE="${LAB_REGISTRY}/aim345-router:0.3.2"
docker build --target engine -t "$LAB_IMAGE" .
docker build --target router -t "$LAB_ROUTER_IMAGE" .
docker run --rm --entrypoint python3 "$LAB_IMAGE" /lab/verify_image.py
docker push "$LAB_IMAGE"
docker push "$LAB_ROUTER_IMAGE"
cp config.example.json config.json
```

Fill `context`, `namespace`, `image`, `router_image`, `instance_type`, `nodes`, and the GPU/EFA counts in `config.json`. `nodes` are Kubernetes hostname labels for resources allocated to this lab. The namespace must start with `aim345-`. Add only the tolerations needed for that allocation. The scripts disable preemption and never scale or delete another workload. A pending pod is a capacity or configuration finding; inspect its events rather than freeing occupied capacity.

The engine requests `80Gi` of ephemeral storage and the CPU router requests `2Gi`. These are storage reservations, not measured application usage. Verify enough free container-image space on each target node before pulling; image downloads and unpacking can exceed the image layer sizes. An earlier Seoul attempt using the engine image for the router exhausted a system-node filesystem. The separate router image and explicit requests address that deployment error.

Model weights are downloaded at the pinned revision into the dedicated `model_cache_host_path` on each node, then mounted at `/models/<revision>`. Allow space for the model snapshot and download cache. The default model is public. A different gated model also needs an authorized download credential, a pinned revision, compatible tokenization, and separate validation. The default BF16 KV format and context limit are explicit configuration assumptions.

To review both configurations before applying them:

```bash
mkdir -p rendered
./0.deploy-unified.sh --config config.json --render > rendered/unified.json
./1.deploy-disaggregated.sh --config config.json --render > rendered/disaggregated.json
```

## Check the example end to end

This block checks that the example works on your allocation, from a fresh clone to cleanup, in one paste. It runs the unit tests, deploys the unified stack and measures shape B at 0.5 and 1 tasks/s for 20 seconds each, does the same for the disaggregated stack, runs `5.collect.py` on both sets of ramp files, and deletes the lab namespace. It takes 10 to 30 minutes on two `g7.48xlarge` nodes, most of it waiting for the model servers to load; the shorter time is with the images and model already cached on the nodes.

Set the variables at the top for your environment. `LAB_CONFIG` is a filled-in `config.json` (see "Prepare the image and configuration"); the block reads the Kubernetes context, the namespace, the instance type and the GPU count per stack (two workers of `gpus_per_worker` each) from it, and your kubeconfig must reach that context. `LAB_REGION` is the AWS Region recorded with each ramp file. `LAB_PORT` is a local port that no other program is listening on. The block stops at the first failing step: another program already listens on `LAB_PORT`, the port-forward exits, the router does not answer `/generate`, or a ramp completed no call. After a failure the lab namespace stays up for inspection; paste the block again from the top to retry (it removes `LAB_WORKDIR`, and each deployment replaces the lab's previous one), or delete the namespace with `python3 deployment.py cleanup --config config.json`.

```bash
export LAB_REPO=https://github.com/awslabs/awsome-distributed-ai.git
export LAB_BRANCH=feat/disagg-gap-verdicts
export LAB_WORKDIR="$HOME/aim345-pr1-test"
export LAB_CONFIG="$HOME/config.json"
export LAB_REGION=us-west-2
export LAB_PORT=18000
cd "$HOME"
bash -euo pipefail <<'LAB'
trap 'kill $(jobs -p) 2> /dev/null || true' EXIT
python3 -c 'import os, socket; socket.socket().bind(("127.0.0.1", int(os.environ["LAB_PORT"])))'
wait_router() {
  for i in $(seq 1 180); do curl -sf --max-time 10 -X POST "http://127.0.0.1:$LAB_PORT/generate" -H 'content-type: application/json' -d '{"input_ids":[1,2,3],"sampling_params":{"max_new_tokens":1}}' > /dev/null && return 0; sleep 2; done
  return 1
}
check_completed() {
  python3 -c 'import json,sys; [sys.exit("no completed calls in " + p) for p in sys.argv[1:] if json.load(open(p))["summary"]["completed_calls"] == 0]' "$@"
}
rm -rf "$LAB_WORKDIR"
git clone -b "$LAB_BRANCH" "$LAB_REPO" "$LAB_WORKDIR"
cd "$LAB_WORKDIR/examples/use-cases/disaggregated-inference-prefill-decode"
python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
python3 -m unittest discover -p "test_*.py"
cp "$LAB_CONFIG" config.json
LAB_CONTEXT="$(python3 -c 'import json; print(json.load(open("config.json"))["context"])')"
LAB_NAMESPACE="$(python3 -c 'import json; print(json.load(open("config.json"))["namespace"])')"
LAB_INSTANCE_TYPE="$(python3 -c 'import json; print(json.load(open("config.json"))["instance_type"])')"
LAB_GPUS="$(python3 -c 'import json; print(2 * json.load(open("config.json"))["gpus_per_worker"])')"
python3 deployment.py unified --config config.json
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/unified-0 --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/unified-1 --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/router --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" port-forward --address 127.0.0.1 service/router "$LAB_PORT:8000" &
PF=$!
wait_router
kill -0 $PF
python3 3.generate-long-context.py --output traffic/b.json
python3 4.ramp.py --traffic traffic/b.json --url "http://127.0.0.1:$LAB_PORT" --architecture unified --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation --gpus "$LAB_GPUS" --rates 0.5 1 --duration-s 20 --output results/unified-b
check_completed results/unified-b/*-measured.json
kill $PF
wait $PF 2> /dev/null || true
python3 deployment.py disaggregated --config config.json
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/prefill-0 --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/decode-0 --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/router --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" port-forward --address 127.0.0.1 service/router "$LAB_PORT:8000" &
PF=$!
wait_router
kill -0 $PF
python3 4.ramp.py --traffic traffic/b.json --url "http://127.0.0.1:$LAB_PORT" --architecture disaggregated --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation --gpus "$LAB_GPUS" --rates 0.5 1 --duration-s 20 --output results/disaggregated-b
check_completed results/disaggregated-b/*-measured.json
kill $PF
wait $PF 2> /dev/null || true
python3 5.collect.py --runs results/unified-b results/disaggregated-b --output results/e2e-collected
ls -l results/e2e-collected/report.html
python3 deployment.py cleanup --config config.json
LAB
```

A successful run ends with the size of `results/e2e-collected/report.html` and a line reporting that the `LAB_NAMESPACE` namespace was deleted. The measurement here is short and only shows that each step works; use "Sequential comparison" for results you intend to compare.

## Paired endpoints on separate allocations

[paired.py](paired.py) checks that both stacks have the same model revision, images, GPU count, transport request, cache settings and placement mode, then renders separate namespaces and nonpreempting PriorityClasses. It rejects overlapping node sets. The sequential scripts remain available for switching a single allocation in place.

Copy the configuration for each stack:

```bash
cp config.example.json config.unified.json
cp config.example.json config.disaggregated.json
```

Fill both files before deployment. Give them different `aim345-*` namespaces and disjoint assigned node lists. Set `placement` to `packed` for one node per stack, or `separate-nodes` for two nodes per stack. Packed placement requires an even allocated GPU count: set `gpus_per_worker` to half that count and `efa_per_worker` to the pod's assigned EFA count. Each engine process receives a disjoint `CUDA_VISIBLE_DEVICES` list. Packed placement uses pod networking, engine TCP ports 30000 and 30001, and a shared EFA device allocation; the prefill bootstrap listener uses TCP port 8998. The CPU router is pinned to the same assigned node. The packed pod defaults to a CPU request of 8 cores and a memory request of 128 GiB; `packed_cpu` and `packed_memory` configure them equally for both stacks.

For the measured `g7e.12xlarge` shape, both configuration files use `gpus_per_worker: 1` and `efa_per_worker: 1`, requesting two GPUs and one EFA device per stack. Keep hardware labels, model settings and resource requests explicit. Model staging and engine compilation happen before the timed participant session.

Render and deploy both stacks from a fresh pair of namespaces:

```bash
python3 paired.py render
python3 paired.py deploy
```

The deploy operation refuses existing namespaces. It does not replace either live stack when measuring the other. Load the endpoint variables from the prepared configuration files. Set the Region to the event's actual Region before collecting measurements:

```bash
export LAB_CONTEXT="$(python3 -c 'import json; print(json.load(open("config.unified.json"))["context"])')"
export LAB_UNIFIED_NAMESPACE="$(python3 -c 'import json; print(json.load(open("config.unified.json"))["namespace"])')"
export LAB_DISAGG_NAMESPACE="$(python3 -c 'import json; print(json.load(open("config.disaggregated.json"))["namespace"])')"
export LAB_INSTANCE_TYPE="$(python3 -c 'import json; print(json.load(open("config.unified.json"))["instance_type"])')"
export LAB_GPUS="$(python3 -c 'import json; print(2 * json.load(open("config.unified.json"))["gpus_per_worker"])')"
export LAB_REGION="$(python3 -c 'import json; print(json.load(open("config.unified.json"))["context"].split(":")[3])')"
```

For packed placement, wait for the engine pod and router in each namespace:

```bash
kubectl --context "$LAB_CONTEXT" -n "$LAB_UNIFIED_NAMESPACE" rollout status deployment/engines --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_UNIFIED_NAMESPACE" rollout status deployment/router --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_DISAGG_NAMESPACE" rollout status deployment/engines --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_DISAGG_NAMESPACE" rollout status deployment/router --timeout=1800s
```

With separate-node placement, wait for the individual worker Deployments named in the rendered files, using the sequential path's readiness commands in each namespace. Run each port-forward command in its own terminal with the same environment:

```bash
kubectl --context "$LAB_CONTEXT" -n "$LAB_UNIFIED_NAMESPACE" port-forward service/router 8000:8000
kubectl --context "$LAB_CONTEXT" -n "$LAB_DISAGG_NAMESPACE" port-forward service/router 8001:8000
```

Verify both HTTP endpoints and their live GPU requests, then inspect the disaggregated engines' selectors, installed distributions and EFA counters around a real request:

```bash
python3 paired.py verify --output results/paired-before
python3 7.verify-transport.py --config config.disaggregated.json --url http://127.0.0.1:8001 --output results/paired-transport.json
```

The packed verifier reads both live engine processes' command lines and selector environments. A packed request transfers state between processes on one node; its result does not establish cross-node EFA transfer. Record the observed counter deltas, including a flat counter, and the report's `cross_node_efa_status`. With separate-node placement, a passing transport check also requires positive RDMA byte deltas. Preserve provider logs with either result.

Generate the same traffic definitions and run each shape against both resident endpoints. The GPU count passed to every run is the count for one stack:

```bash
python3 2.generate-agentic.py --output traffic/a.json
python3 3.generate-long-context.py --output traffic/b.json
python3 4.ramp.py --traffic traffic/a.json --url http://127.0.0.1:8000 --architecture unified --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation --gpus "$LAB_GPUS" --rates 0.1 --duration-s 30 --output results/paired-unified-a
python3 4.ramp.py --traffic traffic/a.json --url http://127.0.0.1:8001 --architecture disaggregated --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation --gpus "$LAB_GPUS" --rates 0.1 --duration-s 30 --output results/paired-disaggregated-a
python3 4.ramp.py --traffic traffic/b.json --url http://127.0.0.1:8000 --architecture unified --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation --gpus "$LAB_GPUS" --rates 0.25 0.5 1 2 4 8 --duration-s 30 --output results/paired-unified-b
python3 4.ramp.py --traffic traffic/b.json --url http://127.0.0.1:8001 --architecture disaggregated --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation --gpus "$LAB_GPUS" --rates 0.25 0.5 1 2 4 8 --duration-s 30 --output results/paired-disaggregated-b
python3 5.collect.py --config config.unified.json --runs results/paired-unified-a results/paired-unified-b --output results/paired-unified-evidence
python3 5.collect.py --config config.disaggregated.json --runs results/paired-disaggregated-a results/paired-disaggregated-b --output results/paired-disaggregated-evidence
python3 5.collect.py --runs results/paired-unified-a results/paired-unified-b results/paired-disaggregated-a results/paired-disaggregated-b --output results/paired-comparison
python3 paired.py verify --output results/paired-after
```

Compare the before/after pod UIDs and GPU requests to establish that both stacks remained resident. The collector saves both packed engines' metrics and server configuration. The existing common SLO and failure-accounting rules apply. A short execution check can use a rate of 0.1 tasks/s and an offered window of 10 seconds for each shape, as recorded in `VALIDATION.md`; that check does not establish sustained capacity. Keep the two prepared endpoints for one-factor crossover repeats, changing the same traffic definition for both arms.

After saving evidence, remove the pair and its PriorityClasses:

```bash
python3 paired.py cleanup
python3 -m unittest -v test_paired test_lab test_scoring test_benchmark test_report test_collect
```

## Sequential comparison

The numbered scripts follow the preparation and experiment order. Scripts numbered after collection are repeatable diagnostic, scaling, and cleanup steps. The deployment scripts switch architectures on the same allocated nodes by removing only this lab's serving resources. Full GPU allocations cannot host both architectures simultaneously on the same GPUs. Rehearse the switch and weight-loading time; simultaneous predeployment requires separate, equal GPU allocations and separate namespaces.

Set convenient controller variables and generate both traffic shapes:

```bash
export LAB_CONTEXT="$(python3 -c 'import json; print(json.load(open("config.json"))["context"])')"
export LAB_NAMESPACE="$(python3 -c 'import json; print(json.load(open("config.json"))["namespace"])')"
export LAB_INSTANCE_TYPE="$(python3 -c 'import json; print(json.load(open("config.json"))["instance_type"])')"
export LAB_GPUS="$(python3 -c 'import json; print(2 * json.load(open("config.json"))["gpus_per_worker"])')"
export LAB_REGION=ap-northeast-2
python3 2.generate-agentic.py --output traffic/a.json
python3 3.generate-long-context.py --output traffic/b.json
```

Shape A is a structural agent simulation: each task makes between 5 sequential calls and 15 sequential calls, retains the actual preceding responses, and receives simulated tool observations under a shared system prompt. It checks between 50 percent and 90 percent token-prefix overlap between consecutive prompts. These are workload settings, not measured server cache-hit rates. Shape B has one call per task and an input length of 8192 tokens by default. It changes the beginning of each document between warmup and measurement to prevent accidental whole-prompt cache hits. Both shapes request 256 output tokens, with EOS stopping disabled to keep output work comparable. The task text is synthetic and does not establish production task quality.

For Round 0, deploy unified and wait for readiness:

```bash
./0.deploy-unified.sh --config config.json
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/unified-0 --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/unified-1 --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/router --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" port-forward service/router 8000:8000
```

Keep port forwarding running in a separate terminal. Run the following controller commands with the virtual environment active. The rate unit is **tasks per second**. For shape B that also means calls per second. For shape A, arrivals remain open-loop at the task boundary while calls within each task remain sequential; the summary reports actual call rate separately.

```bash
python3 4.ramp.py --traffic traffic/a.json --url http://127.0.0.1:8000 --architecture unified --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation --gpus "$LAB_GPUS" --rates 0.1 --duration-s 30 --output results/unified-a
python3 4.ramp.py --traffic traffic/b.json --url http://127.0.0.1:8000 --architecture unified --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation --gpus "$LAB_GPUS" --rates 0.25 0.5 1 2 4 8 --duration-s 30 --output results/unified-b
python3 5.collect.py --config config.json --runs results/unified-a results/unified-b --output results/unified-evidence
```

These rates and durations are starting settings, not observed crossover points. Each rate has a discarded warmup run followed by a measured run. All requests drain before the next rate starts; failures and timeouts remain in the result. Warmup uses the same traffic shape and offered rate. Extend the measurement window and repeat rates in reversed order before describing a rate as sustained.

For Round 1, stop the port-forward command, deploy disaggregation, wait for each engine and router, then restart port forwarding:

```bash
./1.deploy-disaggregated.sh --config config.json
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/prefill-0 --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/decode-0 --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/router --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" port-forward service/router 8000:8000
```

Run shape A with the same settings and GPU budget:

```bash
python3 7.verify-transport.py --config config.json --url http://127.0.0.1:8000 --output results/transport.json
python3 4.ramp.py --traffic traffic/a.json --url http://127.0.0.1:8000 --architecture disaggregated --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation --gpus "$LAB_GPUS" --rates 0.1 --duration-s 30 --output results/disaggregated-a
```

**UNVALIDATED expected result:** a well-cached unified deployment wins TTFT on shape A. Round-robin routing distributes shared prefixes to both replicas but does not guarantee that each task's evolving history stays cached on its next replica. Use measured cached-token counts and engine metrics to explain the result. Do not change the baseline to a single unified replica or quietly change routing policy to obtain the expected ranking.

### The transport-selector trap

Both engines need both settings:

```text
--disaggregation-transfer-backend nixl
SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC
```

The first selector defaults to `mooncake`. The second defaults to `UCX`. Setting only the environment variable leaves SGLang on Mooncake; setting only the CLI flag leaves NIXL on UCX. `--disaggregation-ib-device` is Mooncake-only and must be absent from this NIXL path. The bootstrap server is on prefill, using TCP port `8998` in these configurations. The router receives each prefill endpoint and that bootstrap port explicitly. The pinned [SGLang connector](https://github.com/sgl-project/sglang/blob/v0.5.12.post1/python/sglang/srt/disaggregation/nixl/conn.py) and [environment definitions](https://github.com/sgl-project/sglang/blob/v0.5.12.post1/python/sglang/srt/environ.py) document the inner selector.

The transport script inspects the deployed arguments, installed packages, and EFA RDMA byte counters around a successful streamed request. Preserve the provider and GPU-memory-registration logs as well. Counter movement alone cannot exclude background traffic or prove the absence of host staging. For a hardware selector matrix, restart only this lab's engines for each of these controls, preserve the rendered manifests and logs, and restore the positive configuration afterward:

| SGLang CLI selector | NIXL environment selector | Source-validated selection | Hardware status |
|---|---|---|---|
| Omitted | Omitted | Mooncake | UNVALIDATED |
| Omitted | `LIBFABRIC` | Mooncake; inner NIXL variable unused | OBSERVED, Oregon g7e: ready, then QP creation failure and prefill segmentation fault on request |
| `nixl` | Omitted | NIXL with UCX | OBSERVED, Oregon g7e: ready, then unreachable link-local peer and `NIXL_ERR_BACKEND` on request |
| `nixl` | `LIBFABRIC` | NIXL with LIBFABRIC and `FI_PROVIDER=efa` | VALIDATED, Oregon g7e cross-node: completed request and matching RDMA byte volume; see [validation record](VALIDATION.md) |

### Attribute the handoff cost and prepare connections

**UNVALIDATED expected result:** connection setup and scheduling dominate the extra TTFT for a small compressed KV transfer, as in prior work. Estimate the logical MLA cache as `input_tokens × layer_count × (kv_lora_rank + qk_rope_head_dim) × bytes_per_element`. Include every layer. BF16 uses 2 bytes per element. Then account for tensor-parallel replication, page padding, and the actual transfer layout before comparing total transferred bytes with measured GPU-buffer fabric bandwidth. Per-layer compressed-state size is not the full-model transfer volume.

The included `fabric_probe.py` helper transfers CUDA buffers between the existing lab engine pods using the pinned NIXL library and verifies the destination contents. Run it while inference traffic is idle. In the first terminal, bind the server to the decode node's private address; in the second, connect from the prefill pod. TCP port `19000` is the control channel. Both peers need the same backend and transfer size.

```bash
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" exec deployment/decode-0 -c engine -- python3 /lab/fabric_probe.py server --host "$LAB_DECODE_PRIVATE_IP" --instance-type "$LAB_INSTANCE_TYPE" --eager
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" exec deployment/prefill-0 -c engine -- python3 /lab/fabric_probe.py client --host "$LAB_DECODE_PRIVATE_IP" --instance-type "$LAB_INSTANCE_TYPE" --eager
```

Set `LAB_DECODE_PRIVATE_IP` from the decode pod's `status.podIP`; host networking makes it the node's private address. The helper's defaults are a buffer size of 64 MiB, an iteration count of 20 transfers, and one GPU per peer. Its `warm_gib_per_s` includes Python polling and covers a single GPU buffer, not every tensor-parallel rank. The `--eager` switch exercises NIXL's own `make_connection` API in this independent probe; it does not modify SGLang. Restart both probe processes without that switch for a cold-path comparison. Provider logs and isolated EFA counter changes remain part of interpreting the result.

Provide an actual bandwidth measurement and its transcript to `7.verify-transport.py` with `--bandwidth-gib-s` and `--bandwidth-evidence`. It records the logical byte estimate and ideal wire time in milliseconds. It does not insert a datasheet bandwidth or claim that subtracting wire time alone proves a handshake root cause. Compare cold starts, warmed connections, and steady-state request traces on the same hardware.

The pinned connector calls `add_remote_agent` after peer discovery but contains no explicit eager NIXL `makeConnection` call or supported bootstrap switch for it. The requested engine-level mitigation therefore remains **UNVALIDATED and blocked on upstream support**. This example contains no source patch, monkey patch, or replacement connector.

An operational workaround can prime the fixed prefill/decode pairs before admitting measured traffic:

```bash
python3 8.warm-connections.py --config config.json --url http://127.0.0.1:8000 --output results/startup-prime.json
```

The script issues startup requests through the round-robin router and flushes engine prompt caches afterward. That is request priming, not an implementation of eager `makeConnection`. It also warms kernels, so a timing improvement is not pure connection-setup attribution. A clean cold-versus-primed comparison needs fresh engine starts, identical compilation state, separate prefixes, and explicit labels; routine per-rate warmup already removes some cold costs. The warmup workaround's effect remains **UNVALIDATED**.

## Round 2: useful throughput and independent scaling

Run shape B against disaggregation with the same offered rates and absolute objectives used for unified:

```bash
python3 4.ramp.py --traffic traffic/b.json --url http://127.0.0.1:8000 --architecture disaggregated --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation --gpus "$LAB_GPUS" --rates 0.25 0.5 1 2 4 8 --duration-s 30 --output results/disaggregated-b
python3 5.collect.py --config config.json --runs results/disaggregated-a results/disaggregated-b --output results/disaggregated-evidence
python3 5.collect.py --runs results/unified-a results/unified-b results/disaggregated-a results/disaggregated-b --output results/comparison
```

The starting objectives are TTFT at most 2000 ms and average TPOT at most 100 ms, with joint attainment of at least 90 percent of planned calls. These are configurable lab objectives, not production commitments. Use `--ttft-slo-ms`, `--tpot-slo-ms`, and `--attainment-fraction` to set the same absolute objective for both arms. Do not calibrate separate objectives from each arm's unloaded latency.

**UNVALIDATED expected result:** at some prefill-heavy offered rate, unified violates the latency objective while disaggregation remains within it. Both architectures have finite capacity. The lab leaves chunked prefill and prefix caching enabled in unified; chunked prefill can reduce interference, so a collapse is not guaranteed. Describe the observed ranking if it differs. Inspect queue and cache metrics before attributing a failure to prefill.

When prefill is the binding pool, append a separately allocated third hostname to `nodes` in `config.json`. Preserve all existing engine settings. Then run:

```bash
./6.scale-prefill.sh --config config.json
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/prefill-1 --timeout=1800s
kubectl --context "$LAB_CONTEXT" -n "$LAB_NAMESPACE" rollout status deployment/router --timeout=1800s
export LAB_SCALED_GPUS="$(python3 -c 'import json; print(3 * json.load(open("config.json"))["gpus_per_worker"])')"
```

Restart port forwarding after the router changes. Repeat the shape B ramp with `--architecture disaggregated-scaled`, `--gpus "$LAB_SCALED_GPUS"`, and a new output directory. The scaling script adds a prefill worker and changes router membership while preserving the decode Deployment. Capture decode pod identity and restart count before and after. **UNVALIDATED expected result:** the objective recovers if prefill capacity was the bottleneck. This arm has a larger GPU budget and is not an iso-resource comparison against the original unified baseline. Report useful throughput per GPU; a fair capacity comparison would add a full unified replica, which adds both prefill and decode capacity.

## Read the measurements and locate the crossover

`summary.csv` has one row per measured rate and TPOT definition and records hardware, evidence scope, offered task rate, actual call rate, successful-call throughput, useful-call throughput, GPU-normalized useful throughput, joint SLO attainment, and successful-request latency percentiles. Useful throughput is calls satisfying both latency bounds divided by the full interval including drain. Failed requests, truncated streams, timeouts, and skipped dependent calls contribute no useful work and stay in the attainment denominator. Successful-request percentiles alone can hide failures. Average TPOT comes from server token counts and client stream timing; chunks containing several tokens limit per-token resolution.

The harness never caps concurrent tasks to hide overload. `max_client_launch_lag_ms` detects controller-side scheduling lag; excessive lag marks the rate invalid. For agentic traffic, later calls depend on earlier responses, so the offered task rate and realized call rate differ. The highest tested qualifying rate is a finite-window observation. Extend runs and inspect queue growth before calling it sustained goodput capacity.

For Round 3, edit one factor at a time in a copy of the shape JSON and regenerate traffic: shape B's `input_tokens`; the ramp's `--rates`; or shape A's `shared_system_fraction`, keeping its sequential-call and overlap bounds intact. Lower shared-system availability is an input intervention, not a guaranteed cache-hit percentage. Record the engine's cached-token observations and cache-policy settings. Repeat both architectures at each setting and enter the observed crossover into the [decision framework](DECISION-FRAMEWORK.md).

The session budget is framing for 5 minutes, Round 0 for 12 minutes, Round 1 for 15 minutes, Round 2 for 13 minutes, Round 3 for 8 minutes, and wrap-up for 5 minutes, leaving 2 minutes for transitions. These are teaching allocations. Image preparation, model loading, and the full benchmark sweep belong in rehearsal. Round 3 can use a guided comparison of measurements already collected if time is short.

## Three TPOT definitions

Every ramp records each call's own TTFT, average TPOT, and the arrival time and token count of each chunk a completed call received, so a stored run can be judged under three senses of "TPOT" without new traffic. Gaps are taken only from completed calls; a stream broken partway through says nothing reliable about the gaps inside it.

| Definition | What it judges | Compares with |
|---|---|---|
| `mean` (default) | A request's own averaged TPOT: `(last_chunk_ts - first_chunk_ts) / (output_tokens - first_chunk_tokens)`. Must be `<=` the bound for that request to count as good. | Closest to a simple end-to-end-latency read; most sensitive to a few very fast or very slow responses averaging out. |
| `itl-p99` | Every per-chunk gap of the seed at that rate, divided by its new-token count and repeated once per token -- the same pooled construction `sglang.bench_serving --backend sglang` reports as ITL -- pooled and its p99 checked once per rate. A request only has to pass TTFT to count as good; TPOT is not a per-request test. | Closest to `sglang.bench_serving`'s own ITL metric, with a nearest-rank percentile (`scoring.percentile`) rather than `vllm bench serve`'s interpolated one. |
| `worst-itl` | A request's own single longest gap between two consecutive chunks, not divided by token count: a chunk that batches several tokens after a long stall is one real pause, not several short ones averaged away. Must be `<=` the bound for that request to count as good -- a per-request test, like `mean`, but on the worst gap instead of the average. | Catches a response that stalled badly once even though its average TPOT looked fine; `mean` and `itl-p99` can both pass a request `worst-itl` fails. |

A seed at a rate qualifies only once at least `--attainment-fraction` (default 0.9) of its *planned* calls are good under the chosen definition -- failed, truncated and skipped calls stay in that denominator. A ramp file without per-chunk records, or whose successful calls do not all carry them, is reported as `unavailable` under `itl-p99` and `worst-itl` rather than approximated; `mean` still scores it.

## How a qualifying rate and its goodput are computed

Each rule's own per-call or per-seed test, next to the function that implements it:

| Rule | What a call (or seed) must do to count as good | Function |
|---|---|---|
| `mean` | That call's own averaged TPOT `<=` the bound. | `scoring.summarize`'s `tpot_ok`, reading `tpot_ms` |
| `itl-p99` | Not a per-call test: every per-chunk gap of the seed at that rate, divided by its new-token count, is pooled (`scoring.pooled_itls_ms`) and the pool's nearest-rank p99 (`scoring.percentile`) must be `<=` the bound; a call only needs a passing TTFT to count as good. | `scoring.pooled_itls_ms`, `scoring.percentile` |
| `worst-itl` | That call's own single longest raw gap between two chunks (not divided by token count) `<=` the bound. | `scoring.worst_itl_ms` |

`scoring.summarize`'s `slo_attainment_fraction` is the share of a seed's *planned* calls (not only completed ones) that were good under the chosen rule; a seed at a rate qualifies once that fraction reaches `--attainment-fraction` (default 0.9). A rate qualifies when every seed that ran it qualified and every lower planned rate also qualified (`scoring.highest_qualifying`); a rate the ramp stopped before reaching counts as failing.

Goodput at a qualifying rate is `useful_calls_per_s_per_gpu`: calls that are good under the chosen rule, divided by the measurement window including drain, divided by the GPU count. Under `mean` and `worst-itl` a good call passed TTFT and its own TPOT or longest-gap test; under `itl-p99` every TTFT-passing call counts, at a rate whose pooled p99 met the bound.

## Worked example: three verdicts and goodput by hand

This walks one rate, one seed, through `scoring.py` by hand, using four made-up responses rather than a real measurement. The target in this example is TTFT `<=` 2,000 ms, TPOT `<=` 100 ms, and at least 90% of calls good.

**Four responses.** `benchmark.py` records each chunk's arrival time and the number of new tokens it carried:

| Response | TTFT (ms) | Chunk gaps, with the tokens each chunk carried |
|---|---|---|
| R1 | 500 | 20 ms (1 token) x 4 |
| R2 | 600 | 20 ms (1 token) x 19, then one 640 ms gap (1 token) |
| R3 | 2,500 | 20 ms (1 token) x 4 |
| R4 | 400 | 20 ms (1 token) x 10, then one 200 ms gap (4 tokens) |

R2's 640 ms gap is one chunk arriving late after many fast ones. R4's 200 ms gap arrives as a single chunk carrying four tokens at once -- the stall produced one chunk, not four.

**Step 1 -- is each call good under each rule?** Every rule requires TTFT `<=` 2,000 ms first; only R3 fails that gate. Above that gate, the three rules read TPOT differently:

| Rule | What it reads | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| `mean` | `(last_chunk_ts - first_chunk_ts) / (output_tokens - first_chunk_tokens)`; each response's first chunk carries 1 token here, so this reads `(output_tokens - 1)` | 80/4 = 20 ms, good | (19x20+640)/20 = 51 ms, good | fails TTFT | (10x20+200)/14 ~ 29 ms, good |
| `worst-itl` (`scoring.worst_itl_ms`) | the single longest raw gap, not divided by that chunk's token count | 20 ms, good | 640 ms > 100, fails | fails TTFT | 200 ms > 100, fails |
| `itl-p99` | no per-call test here; TTFT passing is enough. TPOT is judged once per seed, in step 2 | good | good | fails TTFT | good |

`mean` divides R2's 640 ms stall and R4's 200 ms stall across their surrounding fast gaps, so both average out to a passing TPOT. `worst-itl` takes R4's 200 ms gap un-divided, as the one real pause the caller waited through, which is why it fails there even though `mean` passes it.

**Step 2 -- does the seed qualify at this rate?** `scoring.summarize`'s `slo_attainment_fraction` is good calls divided by *planned* calls, including any that failed or never ran, checked against `--attainment-fraction` (default 0.9). Say this seed planned 10 calls at this rate and got back responses shaped like R1 x 6, R2 x 2, R4 x 1, R3 x 1:

| Rule | Good calls | Fraction | Qualifies? |
|---|---|---|---|
| `mean` | R1 x 6 + R2 x 2 + R4 x 1 = 9 | 9/10 = 90% | yes |
| `worst-itl` | R1 x 6 = 6 | 6/10 = 60% | no |

`itl-p99` is judged differently, over the same 10-call seed as the attainment table above (R1 x 6, R2 x 2, R4 x 1, R3 x 1), not a per-call test. `scoring.pooled_itls_ms` pools every chunk gap from every *completed* call in that seed -- R3 counts too, since it finished; only its TTFT failed -- dividing each gap by its chunk's token count and repeating it once per token, the same construction `sglang.bench_serving --backend sglang` reports as ITL. Pooling R1's 6 copies, R2's 2, R4's 1 and R3's 1 gives 76 values of 20 ms, 4 values of 50 ms (R4's 200 ms gap split across its 4 tokens), and 2 values of 640 ms (R2's two calls): 82 values in all. `scoring.percentile`'s nearest-rank p99 sits at position `ceil(82 x 0.99) = 82`, the single largest value, 640 ms -- over the 100 ms bound, so `itl-p99` fails here too. A much larger pool with that same kind of rare stall diluted among far more fast gaps would push it past the 99th-percentile position and let p99 pass: a long stall drops out of the 99th percentile once enough other traffic is pooled with it. An `itl-p99` seed qualifies only when both this pooled p99 and the 90% attainment on TTFT-passing calls hold.

**Step 3 -- does the rate qualify?** `scoring.highest_qualifying` walks rates from lowest to highest and takes the last one where every seed that ran it qualified and every lower rate also qualified; a rate with no file at all, because the ramp stopped early, fails exactly like a rate whose file failed.

| Rate (tasks/s) | seed 345 | seed 1001 | seed 1002 | This rate |
|---|---|---|---|---|
| 0.25 | pass | pass | pass | qualifies |
| 0.5 | pass | pass | pass | qualifies |
| 1 | pass | fail | pass | fails -- walk stops here |
| 2 | pass | pass | pass | not read (a lower rate already failed) |

The qualifying rate here is 0.5, even though every seed passed at rate 2: rate 1 failing underneath it is enough to disqualify it.

**Step 4 -- what goodput looks like, by rule.** `useful_calls_per_s_per_gpu` is good calls at a qualifying rate, divided by the measurement window (including drain), divided by GPU count. Reusing the step-2 counts (9 good under `mean`, 6 under `worst-itl`, 9 TTFT-passing under `itl-p99`) with a 22 s window and 4 GPUs:

| Rule | Good calls | `useful_calls_per_s_per_gpu` |
|---|---|---|
| `mean` | 9 | 9/22/4 ~ 0.102 |
| `worst-itl` | 6 | 6/22/4 ~ 0.068 |
| `itl-p99` | 9 (every TTFT-passing call; not a per-call TPOT subset) | 9/22/4 ~ 0.102 |

On this made-up seed, only `mean` actually qualifies this rate: step 2 showed `worst-itl` at 60% attainment (below the 90% target) and `itl-p99`'s pooled p99 at 640 ms (over the 100 ms bound). The `worst-itl`/`itl-p99` rows above show how the formula computes from the same counts, not a goodput this toy rate actually earned under those two rules -- `5.collect.py`/`report.html` only ever report this figure for a rule a rate qualified under.

**A file with no or incomplete chunk recording.** `itl-p99` and `worst-itl` report such a file as `unavailable` rather than scoring it as zero; `mean` still scores it.

## Score against another objective

`5.collect.py` judges every file at the objective `4.ramp.py` recorded with it. To read the same stored runs at another objective, give any of `--ttft-slo-ms`, `--tpot-slo-ms`, `--attainment-fraction` and `--ttft-metric` (`per-call`, `p95` or `p99`); the values not given stay as recorded. No traffic is sent:

```bash
python3 5.collect.py --runs results/unified-b results/disaggregated-b --tpot-slo-ms 30 --output results/tpot-30
```

`report.html` states whether its objective was recorded with the runs or given on the command line, and its promise maps still cover the whole `--ttft-ms` x `--tpot-ms` grid.

## 5.collect.py's output

`5.collect.py` reads the `*-measured.json` files under each `--runs` directory, scores each one under `mean`, `itl-p99` and `worst-itl` at the objective it was recorded with (or one given on the command line, see "Score against another objective"), and writes `summary.csv` and `qualified-rates.json` into `--output`, plus `report.html` when at least one shape was measured on both `unified` and `disaggregated` with the same seeds and planned rates and `--no-report` is not given. It sends no traffic.

| File | One row or entry is | What it holds | Read it to |
|---|---|---|---|
| `summary.csv` | One measured file (one stack, seed and rate) under one TPOT definition. Two stacks measured at two rates with one seed give 12 rows. | `shape`, `stack`, `seed`, `rate`, `definition`, `source` first; then that row's verdict (`meets_joint_slo`), the share of planned calls that counted as good (`slo_attainment_fraction`) and goodput (`useful_calls_per_s_per_gpu`) -- under `itl-p99` both count every call that passed TTFT, and the pooled p99 enters only the verdict; then the file's identity, the bounds it was judged at, and every latency, gap and validity statistic of that file. | See why one seed at one rate passed or failed under one definition, and trace the row back to its file. Filter on `definition` to read one definition at a time. |
| `qualified-rates.json` | One stack's result for one shape under one TPOT definition, under `by_definition`. | The qualifying rate (`highest_qualifying_offered_tasks_per_s`) and the goodput at that rate (`useful_calls_per_s_per_gpu`); the objective each shape was judged at (`objectives_by_shape`); and the reason for any shape that could not be scored (`refused`). A rate of `null` means the stack qualified at no measured rate, so there is no goodput either. | Get one number per stack and definition: how much load the stack took inside the objective, and how much of it was useful. |
| `report.html` | One page, with one section per shape measured on both `unified` and `disaggregated` with the same seeds and planned rates. | In order: the answer (each stack's qualifying rate and goodput under each definition, in one sentence per definition), what was measured (stacks, shape, rates, seeds, GPUs, instance type, objective), the verdict table, one promise map per definition over the `--ttft-ms` x `--tpot-ms` grid (see "The promise map and its two walls"), and the evidence table with one row per stack, rate and seed. Selecting a promise-map cell shows both stacks' qualifying rate and goodput at that objective. Selecting an underlined term opens its definition, with a diagram for the latency, attainment, qualifying-rate and goodput terms. | See the comparison at a glance, and how the winner changes when the objective is tighter or looser. |

A value starting with `unavailable` (`scoring.UNAVAILABLE`) in an `itl-p99` or `worst-itl` row's verdict, attainment and goodput means the file has no complete per-chunk recording, so the gap-based rule cannot be judged; it is never counted as a failure. A file that cannot be read as a ramp file stops `5.collect.py` before it writes anything. When a file whose stored summary no longer matches its own calls, or a shape, is refused, `5.collect.py` still writes its files for everything else, then exits with a non-zero status that names each refusal. A shape measured on only one stack still appears in `summary.csv` and `qualified-rates.json`, and gets no `report.html` section because there is no second stack to draw a promise map against. How the qualifying rate and goodput are computed is in "How a qualifying rate and its goodput are computed" above.

## The promise map and its two walls

A promise map (the maps in `5.collect.py`'s `report.html`) answers one question at each TTFT/TPOT cell: at this latency bound, which architecture tolerates the higher offered rate before failing the attainment target? It is a screen over the finite windows this lab actually measured, not a confidence bound or a capacity estimate -- re-run a chosen cell with more repeats before trusting it as a production threshold.

Two different things can make a cell say "neither qualifies" or refuse to pick a winner, and they are not interchangeable:

- **Both measured, neither qualifies ("-")**: every architecture was run at that cell's objective and none met the attainment target. The wall is real; raise the bound or lower the rate to find where either one does qualify.
- **Incomparable ("?")**: at least one architecture was never judged under this cell's TPOT definition at all -- usually a file written before per-chunk recording existed, or a shape only measured on one architecture. This is a measurement gap, not evidence that the missing side performs worse.

## Cleanup and local checks

Collect logs before switching deployments or cleaning up. Cleanup deletes only this lab's namespace and its named PriorityClass. Model files remain in the dedicated node cache for reuse; remove that dedicated path through your normal node-maintenance procedure if it is no longer needed. Delete any test ECR repository or test node infrastructure you created separately.

```bash
./9.cleanup.sh --config config.json
python3 -m unittest -v test_lab test_scoring test_benchmark test_report test_collect test_paired
```

Common failures include insufficient scheduled GPU/EFA resources, missing EFA devices, incompatible host drivers, insufficient locked-memory permissions, blocked bootstrap TCP, incomplete image pulls, and version mismatches. Inspect pod events and engine logs first. Never treat a successful HTTP response on an unverified transport as an EFA result.

## Facilitator configuration handoff

The facilitator prepares the two resident endpoints and the controller before the participant session. `facilitator/prepare-config.py` generates matched packed configurations from the two explicitly assigned Kubernetes nodes. It requires immutable engine and router images, derives the even GPU count and EFA request from node inventory, copies the assigned node taints as tolerations, and refuses to overwrite an existing configuration. Review the allocation before deploying.

```bash
python3 facilitator/prepare-config.py --context "$LAB_CONTEXT" --nodes "$LAB_NODES" \
  --engine-image "$LAB_ENGINE_IMAGE" --router-image "$LAB_ROUTER_IMAGE"
python3 paired.py render
```

For an offline configuration review, `--inventory` accepts a saved `kubectl get nodes -o json` result and `--output` selects a new output directory. That path was exercised against the saved two-node Oregon EKS inventory from Phase A. It produced two GPUs per stack and one EFA per stack, matching the earlier packed configuration. This was a configuration check, not a serving measurement. The Workshop Studio facilitator runbook handles the released checkout, controller virtual environment, model staging, participant-role kubeconfig, endpoint verification and cache retention. The current template deployment evidence is recorded in the facilitator runbook.

## Spain preparation and constrained paired result, 2026-09-10

`facilitator/prepare-config.py` can cap each packed stack below the physical node's inventory. The following options describe the Spain allocation; supply the actual context, allocated node names and immutable engine/router images through the required arguments:

```bash
python3 facilitator/prepare-config.py --context "$LAB_CONTEXT" --nodes "$LAB_NODES" --engine-image "$LAB_IMAGE" --router-image "$LAB_ROUTER_IMAGE" --namespace-prefix aim345-spain-paired --gpus-per-stack 4 --efas-per-stack 1 --cpu-per-stack 96 --memory-gib-per-stack 384 --model-cache-host-path /mnt/k8s-disks/0/aim345-models
```

Each stack receives four GPUs and one EFA. Its engine pod receives 92 vCPUs and 376 GiB, reserving 4 vCPUs and 8 GiB for the local router within the combined limit. Init-container requests and limits are both 2 vCPUs and 4 GiB. The helper validates the selected GPU/EFA budgets against allocatable resources; without explicit GPU/EFA options it uses all allocatable devices. The current helper derives its default combined CPU/memory budget from the smaller allocatable quantities across the pair, leaving 3 vCPUs and 4 GiB of headroom for node agents. A `/mnt/` host path must be on the intended prepared filesystem; the helper does not mount or provision storage.

On the g7.24xlarge-stand-in allocation of two physical g7.48xlarge nodes, the complete paired DeepSeek-V2-Lite-Chat flow took 27.85 minutes with unchanged engine pod identities. The long-context sweeps plus collection took about 25 minutes, so the participant introduction now reserves 26 minutes for the load-ramp module. An offered window of 30 seconds does not include warmup or request drain. Unified serving qualified at a highest tested rate of 2 tasks/s, while disaggregation qualified at 0.25 tasks/s; no qualifying disaggregated crossover was observed.

DeepSeek-V4-Flash weights alone exceeded the four-GPU node budget by 21.22 GiB. Stock SGLang version 0.5.19 served it with all eight GPUs on one node, and an experimental image with NIXL version 1.4.1 served disaggregation across all sixteen physical GPUs. The pinned SGLang version 0.5.12.post1 failed during scale transformation. These experiments do not change the companion's model or package pins. [VALIDATION.md](VALIDATION.md) preserves exact engine digests, errors, memory arithmetic, cold request latency and experimental transport evidence.

## Record the assigned EKS inventory

```bash
export LAB_NODES="$(python3 -c 'import json; print(" ".join(json.load(open("config.unified.json"))["nodes"] + json.load(open("config.disaggregated.json"))["nodes"]))')"
kubectl --context "$LAB_CONTEXT" get nodes $LAB_NODES -o 'custom-columns=NODE:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,GPUS:.status.allocatable.nvidia\.com/gpu,EFAS:.status.allocatable.vpc\.amazonaws\.com/efa'
```

## Optional: DeepSeek-V4-Flash on g7.48xlarge

DeepSeek-V2-Lite-Chat remains the common main line on both G7 shapes with SGLang version `0.5.12.post1` and NIXL version `1.1.0`. The optional V4 profile uses [Dockerfile.v4](Dockerfile.v4), SGLang version `0.5.19`, both NIXL distributions at version `1.4.1`, and EFA installer version `1.47.0`. Its amd64 engine manifest is `sha256:7da39d58804be6c781991dad9c099bf65c95c49eec42b3d5e927d89105b2e061`; its image index is `sha256:30fd3480469d257011feba3548ee30ed866ebfb1c3fee50f35252300651967b3`. The separately pinned base image and build-time package/source assertions are written in the Dockerfile. The main-line Dockerfile and pins remain unchanged.

The model revision in [config.v4.example.json](config.v4.example.json) contains `159617149040 B`, or approximately `148.7 GiB`, of weight shards. Each observed GPU reports `32623 MiB`. A `g7.48xlarge` node supplies approximately `254.9 GiB` across eight GPUs; a `g7.24xlarge` node supplies approximately `127.4 GiB` across four GPUs. The weights alone exceed the secondary node’s GPU memory by approximately `21.2 GiB`. Each prefill/decode replica needs its own weights. The optional path therefore requires the full primary node pair, prefill TP size of eight ranks on one node and decode TP size of eight ranks on the other, with both EFA devices per node.

Prepare the optional exercise after saving and cleaning up the resident main-line pair. Build the engine separately and preserve its immutable digest:

```bash
docker build --target engine -f Dockerfile.v4 -t "$LAB_V4_IMAGE" .
docker run --rm --entrypoint python3 "$LAB_V4_IMAGE" /lab/verify_image.py
cp config.v4.example.json config.v4.json
```

The facilitator fills the context, node names, router digest and image registry in `config.v4.json`, retaining its optional model and engine profile. Stage approximately 160 GB of weights plus download/compilation space on each node’s NVMe filesystem, outside the timed session. The default example requests 188 vCPUs and 672 GiB per engine, below the observed Spain Kubernetes allocatable limits. Read the live limits before deploying elsewhere. Preserve the existing pinned snapshot and compilation cache; do not replace the main-line cache. `deployment.py` rejects the secondary shape, packed placement and partial GPU/EFA requests for this profile.

```bash
python3 deployment.py disaggregated --config config.v4.json
```

Wait for prefill, decode and router readiness using the rendered deployment names. The package verifier accepts only the two explicitly qualified version tuples and reports `mainline` or `v4-optional`; the transport verifier also checks that the observed profile matches the configuration. The optional participant commands are:

```bash
kubectl --context "$(python3 -c 'import json; print(json.load(open("config.v4.json"))["context"])')" get nodes $(python3 -c 'import json; print(" ".join(json.load(open("config.v4.json"))["nodes"]))') -o 'custom-columns=NODE:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,GPUS:.status.allocatable.nvidia\.com/gpu,EFAS:.status.allocatable.vpc\.amazonaws\.com/efa'
python3 deployment.py disaggregated --config config.v4.json --render > results/v4-rendered.json
```

```bash
export LAB_V4_MODEL="$(python3 -c 'import json; print(json.load(open("config.v4.json"))["model_id"])')"
export LAB_V4_REVISION="$(python3 -c 'import json; print(json.load(open("config.v4.json"))["model_revision"])')"
export LAB_V4_GPUS="$(python3 -c 'import json; print(2 * json.load(open("config.v4.json"))["gpus_per_worker"])')"
export LAB_V4_CONTEXT="$(python3 -c 'import json; print(json.load(open("config.v4.json"))["context"])')"
export LAB_V4_NAMESPACE="$(python3 -c 'import json; print(json.load(open("config.v4.json"))["namespace"])')"
export LAB_V4_INSTANCE_TYPE="$(python3 -c 'import json; print(json.load(open("config.v4.json"))["instance_type"])')"
export LAB_V4_REGION="$(python3 -c 'import json; print(json.load(open("config.v4.json"))["context"].split(":")[3])')"
```

```bash
kubectl --context "$LAB_V4_CONTEXT" -n "$LAB_V4_NAMESPACE" port-forward service/router 8010:8000
```

```bash
python3 7.verify-transport.py --config config.v4.json --url http://127.0.0.1:8010 --output results/v4-transport.json
```

```bash
python3 2.generate-agentic.py --model "$LAB_V4_MODEL" --revision "$LAB_V4_REVISION" --output traffic/v4-a.json
python3 3.generate-long-context.py --model "$LAB_V4_MODEL" --revision "$LAB_V4_REVISION" --output traffic/v4-b.json
```

```bash
python3 4.ramp.py --traffic traffic/v4-a.json --url http://127.0.0.1:8010 --architecture disaggregated --instance-type "$LAB_V4_INSTANCE_TYPE" --region "$LAB_V4_REGION" --evidence-scope mechanism-validation --gpus "$LAB_V4_GPUS" --rates 0.1 --duration-s 30 --output results/v4-a
python3 4.ramp.py --traffic traffic/v4-b.json --url http://127.0.0.1:8010 --architecture disaggregated --instance-type "$LAB_V4_INSTANCE_TYPE" --region "$LAB_V4_REGION" --evidence-scope mechanism-validation --gpus "$LAB_V4_GPUS" --rates 0.1 --duration-s 30 --output results/v4-b
```

```bash
python3 4.ramp.py --traffic traffic/v4-b.json --url http://127.0.0.1:8010 --architecture disaggregated --instance-type "$LAB_V4_INSTANCE_TYPE" --region "$LAB_V4_REGION" --evidence-scope mechanism-validation --gpus "$LAB_V4_GPUS" --rates 0.25 0.5 1 2 --duration-s 30 --output results/v4-sweep
python3 5.collect.py --config config.v4.json --runs results/v4-a results/v4-b results/v4-sweep --output results/v4-evidence
```
