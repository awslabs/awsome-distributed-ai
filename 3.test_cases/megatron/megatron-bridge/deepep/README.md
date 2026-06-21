<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Vendored DeepEP-on-EFA build script

`setup_deepep_efa.sh` here is a **verbatim copy** of
[`micro-benchmarks/expert-parallelism/deepep-benchmark/setup_deepep_efa.sh`](../../../../micro-benchmarks/expert-parallelism/deepep-benchmark/setup_deepep_efa.sh).
It is vendored into this directory only so the shared `Dockerfile` can `COPY` it when
built with `--build-arg EP_BACKEND=nvshmem` — Docker cannot `COPY` files from outside
the build context, and the build context is this test-case directory.

It builds **NVIDIA DeepEP over NVSHMEM-libfabric for AWS EFA** (host-proxy, IBGDA off)
via four source patches: a combined put+signal, an IBGDA→host-proxy shim, fake IBGDA
device state, and `NVSHMEM_MAX_TEAMS` 7→8. See that benchmark's
[README](../../../../micro-benchmarks/expert-parallelism/deepep-benchmark/README.md) for
the full rationale.

**Keep in sync with the canonical copy.** If the benchmark script changes, re-copy:

```bash
cp ../../../../micro-benchmarks/expert-parallelism/deepep-benchmark/setup_deepep_efa.sh \
   setup_deepep_efa.sh
```
