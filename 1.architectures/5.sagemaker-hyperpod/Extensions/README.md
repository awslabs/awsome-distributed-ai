# SageMaker HyperPod Slurm Cluster Extensions

Extensions run during HyperPod Slurm cluster provisioning via the
`OnInitComplete` lifecycle hook. Unlike traditional Lifecycle Configuration
(LCS), extensions plug into the "None (Run extensions)" option in the
HyperPod console and let you compose the cluster with reusable modules
instead of a monolithic LCS bundle.

## Available extensions

| Extension | Purpose |
|-----------|---------|
| [`detect-node/`](./detect-node/) | Writes `/opt/ml/config/nodeinfo.json` identifying the current node as controller / login / compute. Required by any extension that behaves differently per node role. |
| [`add-users/`](./add-users/) | Creates POSIX users, home directories on the shared filesystem, SSH keypairs for inter-node SSH, and Slurm accounting entries. Users are declared in a `shared_users.txt` (CSV) or `shared_users.yaml` file. |
| [`observability/`](./observability/) | Installs node exporter, DCGM exporter, EFA exporter, Slurm exporter, and an OpenTelemetry collector that forwards metrics to Amazon Managed Prometheus. |

`run_extensions.sh` is the orchestrator that chains these together. Its
`ENABLE_ADD_USERS` / `ENABLE_OBSERVABILITY` flags decide what actually
runs; `detect-node` always runs first.

## `prepare_extensions.sh` — helper for staging and uploading

`prepare_extensions.sh` builds the S3 bucket contents that the HyperPod
console needs. It picks which extensions to include, generates the
per-extension config files from your input, patches `run_extensions.sh`
when needed, and uploads everything.

### What it does

1. **Resolves an S3 bucket.** Either uses an existing bucket you name, or
   creates a new one (versioning + public-access-block are set on new
   buckets).
2. **Validates the bucket.** Client-side name check first (S3 naming
   rules), then a live check to distinguish "doesn't exist" from "exists
   but not yours" from "exists in wrong region" — with actionable error
   messages for each.
3. **Generates `shared_users.txt`** from `--users alice,bob,carol` (UIDs
   auto-assigned from 2001) or `--users` + `--uids`, or from an
   interactive prompt. You never need to hand-write the file.
4. **Patches `observability/config.json`** with your Amazon Managed
   Prometheus `remote_write` URL.
5. **Chooses the right entrypoint** based on what you asked for:
   - `--observability` alone → uploads only `observability/`; entrypoint
     is `observability/setup_observability.sh`.
   - `--add-users` alone or with `--observability` → uploads `detect-node/`,
     the selected extension dir(s), and a patched `run_extensions.sh`;
     entrypoint is `run_extensions.sh`.
6. **Prints the full `s3://…` path** to paste into the HyperPod console.

Nothing in the repo is mutated — everything is assembled in a temp
staging dir and uploaded from there.

### Flags

| Flag | Purpose |
|------|---------|
| `--add-users` | Include the add-users extension |
| `--observability` | Include the observability extension |
| `--users u1,u2,u3` | Comma-separated usernames (with `--add-users`). UIDs auto-assigned from 2001 unless `--uids` is given. |
| `--uids 2001,2002,2003` | Explicit UIDs; count must match `--users` |
| `--users-file <path>` | Use a pre-made `shared_users.txt` or `shared_users.yaml` instead of `--users` |
| `--amp-url <url>` | Prometheus `remote_write` URL for observability |
| `--bucket <name>` | Use an existing S3 bucket |
| `--create-bucket <name>` | Create a new S3 bucket |
| `--prefix <path>` | Object key prefix inside the bucket (default: `hyperpod-extensions`) |
| `--region <aws-region>` | AWS region (default: from AWS CLI config) |
| `--aws-profile <name>` | AWS CLI profile to use for every AWS call the script makes |
| `--dry-run` | Print `aws` commands instead of executing them |
| `--yes`, `-y` | Skip interactive prompts where possible |
| `-h`, `--help` | Show help |

### Examples

**Add users only, with a new bucket:**

```bash
./prepare_extensions.sh \
  --add-users \
  --users alice,bob,carol \
  --create-bucket my-hyperpod-extensions-$(date +%s) \
  --region us-west-2
```

**Observability only, existing bucket:**

```bash
./prepare_extensions.sh \
  --observability \
  --amp-url "https://aps-workspaces.us-west-2.amazonaws.com/workspaces/ws-XXXX/api/v1/remote_write" \
  --bucket my-bucket \
  --region us-west-2
```

**Both, non-interactive, using a named AWS profile:**

```bash
./prepare_extensions.sh \
  --add-users --observability --yes \
  --users alice,bob \
  --amp-url "https://aps-workspaces.us-west-2.amazonaws.com/workspaces/ws-XXXX/api/v1/remote_write" \
  --bucket my-bucket \
  --region us-west-2 \
  --aws-profile prod
```

**Interactive (no flags beyond `--add-users`):**

```bash
./prepare_extensions.sh --add-users
# Prompts for: bucket choice, region, usernames, optional UIDs, and confirmation.
```

### Using the output

After a successful upload the script prints something like:

```
Paste this into the HyperPod console
  Custom setup -> Lifecycle configuration -> None

  Entrypoint (full S3 path):
    s3://my-bucket/hyperpod-extensions/run_extensions.sh

  Extensions bucket URI (if the console asks for it separately):
    s3://my-bucket/hyperpod-extensions/
```

Paste the entrypoint path into the HyperPod cluster-creation console
under **Custom setup → Lifecycle configuration → None**, and provisioning
will run the selected extensions on every node.

### Prerequisites

- `aws` CLI installed and configured (or a valid `--aws-profile`)
- `python3` on `PATH` (used for JSON manipulation and YAML parsing)
- IAM permissions for `s3:CreateBucket` (if using `--create-bucket`),
  `s3:PutObject`, `s3:GetBucketLocation`, and `s3:ListBucket` on the
  target bucket
- The HyperPod cluster's execution role must have `s3:GetObject` on the
  extensions prefix so nodes can download the scripts at provisioning
  time
