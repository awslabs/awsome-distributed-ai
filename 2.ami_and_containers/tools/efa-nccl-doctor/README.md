# EFA/NCCL Doctor

A Bash linter for Dockerfiles that build **EFA-enabled distributed-training
containers**. It reports version drift and known-bad configuration
combinations *before* you spend 20 minutes on a build and a multi-node
allocation discovering the problem at runtime.

Runs fully offline by default. No AWS credentials required.

## Why this helper?

Getting EFA working inside a container means getting several independent
things right at once:

- `efa-installer` version
- `aws-ofi-nccl` version — and its confusing tag naming (`1.12.1-aws` vs
  `v1.13.2-aws` vs `v1.19.0`; upstream dropped the `-aws` suffix at v1.14.0)
- NCCL version compatibility with both of the above
- the three container-only `efa_installer.sh` flags
  (`--skip-kmod --no-verify --skip-limit-conf`) that must **not** be used on a host
- removing the bundled HPC-X OMPI/NCCL plugin that otherwise shadows EFA
- `LD_LIBRARY_PATH` ordering so libfabric resolves to `/opt/amazon/efa/lib`

Every one of these fails *silently* in the way that matters: the image builds,
the job launches, NCCL quietly falls back to TCP. You find out later as
"why is my 32-node run slower than 8 nodes?"

This linter encodes those rules as static checks so the failure surfaces at
lint time instead of at scale.

## Prerequisites

1. **bash 4+**, plus `awk`, `sed`, `grep`, `sort` — present on Amazon Linux
   2023, Ubuntu, and macOS with a Homebrew bash.
2. **curl** — only for `--check-latest`.
3. **jq** — only for `--json` and `--check-latest`.

No AWS credentials and no network access are needed for the default mode.

## Usage

```
./efa-nccl-doctor.sh [OPTIONS] <PATH> [<PATH>...]
```

`<PATH>` is a Dockerfile, or a directory scanned recursively for
`Dockerfile`, `Dockerfile.*`, and `*.dockerfile`.

Run `./efa-nccl-doctor.sh --help` for the built-in help.

### Options

| Flag | Description | Default |
|---|---|---|
| `--check-latest` | Resolve the newest `aws-ofi-nccl` release from GitHub and report how far behind each file is. Informational only; never fails the run. | off |
| `--inventory` | Print a version inventory table across all scanned files instead of per-file findings. | off |
| `--strict` | Treat warnings as errors (exit 1 on any warning). | off |
| `--json` | Emit machine-readable JSON on stdout. Requires `jq`. | off |
| `--no-color` | Disable ANSI color (also honors `NO_COLOR`). | off |
| `-h`, `--help` | Show help and exit. | — |
| `-V`, `--version` | Print version and exit. | — |

## Checks

| Code | Level | What it catches |
|---|---|---|
| `EFA001` | error/warn | `efa-installer` not pinned, or pulled via the `latest` tarball |
| `EFA002` | error | `efa-installer` < 1.29.1 without `FI_EFA_SET_CUDA_SYNC_MEMOPS=0` |
| `EFA003` | error | `efa_installer.sh` missing `--skip-kmod` or `--no-verify` (build will fail) |
| `EFA004` | warn | `efa_installer.sh` missing `--skip-limit-conf` |
| `NCCL001` | warn | `aws-ofi-nccl` referenced but version not pinned |
| `NCCL002` | error | `aws-ofi-nccl` tag suffix wrong for its version |
| `NCCL003` | error | Bundled HPC-X OMPI/NCCL plugin not removed (shadows EFA) |
| `NCCL004` | warn | `aws-ofi-nccl` built without `--with-libfabric=/opt/amazon/efa` |
| `ENV001` | error | EFA installed but `/opt/amazon/efa/lib` not on `LD_LIBRARY_PATH` |
| `ENV002` | warn | Bundled OMPI removed but `OPAL_PREFIX` not cleared |
| `PIN001` | warn | Base image uses a floating tag |

### The two that catch the most real bugs

**`NCCL002` — the `-aws` suffix trap.** Upstream `aws-ofi-nccl` tags were named
`v1.7.3-aws` through v1.13.x, then dropped the suffix at v1.14.0. A version
bump that copies the old naming forward (`v1.19.0-aws`) fails at
`git clone -b` with *"Remote branch not found"*, and the reverse
(`v1.7.3` without the suffix) fails the same way. This is the single most
common copy-forward error in this class of Dockerfile.

**`NCCL003` — HPC-X shadowing.** `nvcr.io/nvidia/pytorch` and
`nvcr.io/nvidia/nemo` images bundle HPC-X with its own OpenMPI and
`nccl_rdma_sharp_plugin`. Left in place, they shadow the EFA stack and NCCL
uses the bundled plugin instead. Nothing errors — you just lose EFA.

## Examples

### 1. Lint a single Dockerfile

```bash
./efa-nccl-doctor.sh ../../containers/pytorch/0.nvcr-pytorch-aws.dockerfile
```

### 2. Scan a whole repository

```bash
./efa-nccl-doctor.sh /path/to/awsome-distributed-ai
```

```
--- 4.validation_and_observability/2.gpu-cluster-healthcheck/kubernetes/Dockerfile
  ERROR Dockerfile:34 [EFA001]
        efa-installer pulled via 'latest' tarball -- builds are not reproducible.
        -> Pin ENV EFA_INSTALLER_VERSION=<x.y.z> and interpolate it into the URL.
  ERROR Dockerfile:31 [ENV001]
        EFA installed but /opt/amazon/efa/lib is not on LD_LIBRARY_PATH.
        -> Add: ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH -- otherwise
           libfabric resolves elsewhere and EFA is unused.

Scanned 36 file(s): 3 error(s), 16 warning(s)
```

### 3. Version drift inventory

```bash
./efa-nccl-doctor.sh --inventory /path/to/awsome-distributed-ai
```

```
FILE                                                       EFA_INSTALLER  AWS_OFI_NCCL
---------------------------------------------------------- -------------- ----------------
...containers/pytorch/0.nvcr-pytorch-aws.dockerfile        1.35.0         1.12.1-aws
...3.test_cases/megatron/megatron-bridge/Dockerfile        1.48.0         v1.19.0
...3.test_cases/megatron/nemo/Dockerfile                   1.48.0         v1.19.0
...5.nsight/EKS/Dockerfile.llama2-efa                      1.29.1         v1.7.3-aws

Distinct EFA_INSTALLER_VERSION values:
      1 1.29.1
      1 1.35.0
      1 1.37.0
      6 1.47.0
      4 1.48.0
      2 1.49.0
```

Drift across many files is not automatically wrong — test cases may
legitimately pin different versions. It *is* worth a look when the spread is
wide, because it usually means files were copied forward and never revisited.

### 4. CI gate

```bash
./efa-nccl-doctor.sh --strict 3.test_cases/
```

Exits 1 on any error, or on any warning with `--strict`.

### 5. Machine-readable output

```bash
./efa-nccl-doctor.sh --json . | jq '.findings[] | select(.level=="ERROR")'
```

```json
{
  "scanned": 36,
  "errors": 3,
  "warnings": 16,
  "latest_aws_ofi_nccl": null,
  "findings": [
    {
      "level": "ERROR",
      "file": "...kubernetes/Dockerfile",
      "line": 31,
      "code": "ENV001",
      "message": "EFA installed but /opt/amazon/efa/lib is not on LD_LIBRARY_PATH.",
      "remedy": "Add: ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH ..."
    }
  ]
}
```

### 6. How far behind upstream?

```bash
./efa-nccl-doctor.sh --check-latest 3.test_cases/megatron/
```

Resolves the newest non-prerelease `aws-ofi-nccl` tag from the GitHub API and
emits `NCCL900` INFO findings. **Informational only** — upgrade deliberately
and re-run your NCCL tests; don't chase releases in CI.

## Suppressing a finding

Some deviations are deliberate and verified. `nemo:26.04.01` links torch's
`libtorch_global_deps.so` against `libmpi.so.40` in `/opt/hpcx/ompi/lib`, so
removing HPC-X there breaks `import torch` — `NCCL003` is a true positive but
the *wrong* advice for that file.

Add an inline pragma, in the spirit of `# shellcheck disable=`:

```dockerfile
# efa-nccl-doctor: disable=NCCL003 reason=nemo 26.04.01 torch links libmpi.so.40 from /opt/hpcx/ompi
FROM nvcr.io/nvidia/nemo:26.04.01
```

The `reason=` text is for humans; the linter only matches the code. A linter
that can't be silenced with a documented reason gets ignored wholesale, which
is worse than one with escape hatches.

## Design notes

**Comment-aware.** Dockerfiles in this repository carry long explanatory
comment blocks that routinely mention images and flags they deliberately do
*not* use ("we avoid `nvcr.io/nvidia/pytorch` because..."). All content checks
run against a decommented body, so a file that merely discusses EFA in a
comment is correctly skipped rather than flagged.

**Deterministic regardless of invocation.** `body_has()` matches with a
here-string rather than `decomment "$f" | grep -q ...`. A pipeline ending in
`grep -q` can consume a file descriptor left open by
`while read < <(find ...)`, which made the same file lint differently
depending on whether it was passed directly or discovered by a directory scan.
The selftest asserts both paths agree.

**Conservative by design.** The linter flags patterns known to cause silent TCP
fallback or build failure — not stylistic preferences. Floor values, not
"latest": it never tells you to chase the newest release, only when a
combination is known to misbehave.

## Testing

```bash
./selftest.sh
```

Builds synthetic Dockerfiles in a temp directory and asserts each check fires,
that clean files stay silent, that comment-only mentions don't trigger checks,
that the pragma suppresses, and that the exit-code contract holds.

```
passed: 19   failed: 0
```

Run it after any change to the linter.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | No errors (warnings allowed unless `--strict`) |
| 1 | One or more errors, invalid arguments, or missing prerequisites |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `--json requires jq on PATH` | `jq` not installed | `brew install jq` / `apt-get install jq` |
| `WARNING: could not resolve latest aws-ofi-nccl release` | Offline, or GitHub API rate-limited | Non-fatal; the run continues without `NCCL900` |
| A check fires on a file you know is correct | Deliberate, verified deviation | Add a `# efa-nccl-doctor: disable=<CODE> reason=...` pragma |
| `No Dockerfiles found under: ...` | Path has no matching filenames | The scanner matches `Dockerfile`, `Dockerfile.*`, `*.dockerfile` |
| Findings differ between two runs | Should not happen — see Design notes | Run `./selftest.sh`; open an issue with the file |

## References

- [EFA cheatsheet](../../../1.architectures/efa-cheatsheet.md) — in this repository
- [Reference container recipe](../../containers/pytorch/0.nvcr-pytorch-aws.dockerfile) — the canonical EFA Dockerfile
- [aws-ofi-nccl releases](https://github.com/aws/aws-ofi-nccl/releases)
- [EFA documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
