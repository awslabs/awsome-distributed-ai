# Decide from effective prompt length, latency, and load

Use measurements from the same model revision, GPU budget, cache policy, routing policy, hardware type, and absolute latency objective. Record the instance type and whether the run validates a mechanism or calibrates production. The recommendations below are conditional; every crossover for this lab remains **UNVALIDATED** until measured.

| Question | Observation to record | Implication | Next comparison |
|---|---|---|---|
| How long is the effective prompt after prefix-cache hits? | Input tokens, reported cached tokens, shared-prefix fraction, and routing policy | Short effective prefills leave little compute work for a separate prefill pool to absorb | Compare cached agentic traffic on two unified replicas with equal-GPU disaggregation |
| How tight is the latency objective? | Common TTFT bound in ms, TPOT bound in ms, and required attainment in percent | Handoff overhead can consume a tight TTFT budget; isolated decode can help a tight TPOT budget | Compare joint attainment, including failures, alongside successful-request percentiles |
| Does the offered load create a growing queue? | Offered tasks/s, actual calls/s, queue metrics, and useful calls/s/GPU | Prefill-heavy load may benefit from independent pools; disaggregation also saturates | Ramp long-context traffic, repeat near the qualifying-rate boundary, and extend duration |
| Is prefill the binding pool? | Prefill queue and GPU activity while decode retains spare capacity | An additional prefill worker may recover the objective without adding decode capacity | Preserve the decode pod and add prefill alone; label the increased GPU budget |
| Is the measured handoff penalty mostly wire time? | Full-model KV bytes, TP replication, measured GPU-buffer GiB/s, and cold/warm traces | A small ideal wire time makes bandwidth alone an insufficient explanation | Investigate peer setup and scheduling; upstream eager bootstrap support is still required by this lab |
| Is the transport evidence sufficient? | Both selectors, installed NIXL version, provider logs, GPU registration, and cross-node RDMA byte deltas | A configuration label or aggregate byte counter alone does not establish GPUDirect RDMA | Keep ambiguous runs UNVALIDATED and capture missing evidence |

Record the observed crossover here after a paired run:

| Record | Value |
|---|---|
| Instance type, region, GPU count, model revision, evidence scope | |
| Prompt tokens and measured cached tokens | |
| Offered tasks/s and realized calls/s | |
| TTFT objective in ms, TPOT objective in ms, attainment target in percent | |
| Unified useful calls/s/GPU and joint attainment in percent | |
| Disaggregated useful calls/s/GPU and joint attainment in percent | |
| Queue stability, repeated-run range, and any unresolved transport evidence | |

Choose the topology that meets the workload's objective with the required capacity. Revisit the choice when prefix reuse, effective prompt length, or burstiness changes.
