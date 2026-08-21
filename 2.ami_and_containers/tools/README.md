# Tools

Small, self-contained helpers for building and operating distributed AI
infrastructure on AWS. Each tool lives in its own subdirectory with a README,
has no build step, and is runnable directly from a checkout.

## Available tools

| Tool | Purpose | Needs AWS credentials? |
|---|---|---|
| [`ec2md/`](./ec2md/) | Walk the EC2 IMDSv2 metadata tree and export every value as a shell variable, including flattened JSON. Replaces the `ec2-metadata` CLI. Also imports, exports, and deletes user-data, and toggles IMDS on the local instance. | Only for `--on`/`--off`/`--user-data-import` |
| [`sagemaker_ftp/`](./sagemaker_ftp/) | Find available SageMaker Flexible Training Plan (FTP) offerings in a given Availability Zone, sweeping reservation durations, and recommend the lowest effective `$/instance/hour`. Never purchases. | Yes (`sagemaker:SearchTrainingPlanOfferings`) |
| [`efa-nccl-doctor/`](./efa-nccl-doctor/) | Lint Dockerfiles that build EFA-enabled training containers for version drift and known-bad EFA/NCCL combinations, before a build and a multi-node allocation. Offline by default. | No |

## When you'd reach for each

**Before you provision** — `sagemaker_ftp/find-ftp.sh` tells you what reserved
capacity exists right now and what it costs, without committing to a
non-refundable purchase.

**Before you build** — `efa-nccl-doctor/efa-nccl-doctor.sh` catches the EFA and
`aws-ofi-nccl` misconfigurations that build cleanly, launch cleanly, and then
silently fall back to TCP at scale.

**Once you're on the instance** — `ec2md/ec2md.sh` gets you every piece of
instance metadata as shell variables in one call, which is handy inside
user-data scripts, Slurm prologs, and container entrypoints.

## Conventions

Tools in this directory follow the same shape:

- A single Bash script plus a `README.md`. No packaging, no install step.
- `--help` and `--version` on every tool.
- Prerequisites checked up front with a clear error, not a stack trace.
- Errors and status messages to **stderr**, so `stdout` stays pipeable.
- A `--json` mode where the output is worth machine-reading.
- Read-only by default. Anything that mutates state (purchasing capacity,
  changing IMDS settings) either prints the command for you to run or is
  behind an explicit flag.
- Versions pinned, never `latest`, per the repository
  [CONTRIBUTING](../../CONTRIBUTING.md) guidance.

## Contributing a tool

Add a subdirectory with your script and a `README.md` covering prerequisites,
usage, examples, exit codes, and troubleshooting — see
[`sagemaker_ftp/README.md`](./sagemaker_ftp/README.md) for the reference shape.
Then add a row to the table above.

If your tool has non-obvious behavior worth protecting, ship a `selftest.sh`
alongside it; [`efa-nccl-doctor/selftest.sh`](./efa-nccl-doctor/selftest.sh)
is an example.
