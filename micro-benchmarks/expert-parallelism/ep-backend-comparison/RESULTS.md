# EP Backend Comparison Results on B200

Status: `NOT_RUN_PROFILE_MATRIX_PENDING`

The previous Decode-only result has been retired. It must not be combined with the new two-profile matrix because Prefill-like uses the normal high-throughput API and a different timing boundary.

The replacement campaign will score these workload profiles separately:

| Profile | Tokens | API class | Primary metric |
|---|---:|---|---|
| Decode-like | 128 tokens/rank | Low-latency dispatch and combine | Slowest-rank latency, in ms |
| Prefill-like | 4,096 tokens/rank | Normal high-throughput dispatch and combine, including required layout | Effective logical throughput, in GB/s/rank |

Both profiles use hidden size 7,168, 256 experts, top-k 8 experts/token, FP8 and BF16 dispatch, BF16 combine, EP16 and EP32, 20 warmup iterations, 100 measured iterations, and 3 independent process starts per cell.

No backend performance conclusion is reported until all 72 scored dtype results pass validation and the owned EKS resources pass teardown verification.
