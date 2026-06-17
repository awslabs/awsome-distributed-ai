# SageMaker FTP Finder

A small Bash helper that searches for available
**Amazon SageMaker Flexible Training Plan (FTP)** offerings, scoped to a
specific Availability Zone, and recommends the offering with the lowest
effective `$/instance/hour`.

Supports both:

- **SageMaker HyperPod clusters** (Slurm or EKS) — default
- **SageMaker training jobs** — via `--target training-job`

It wraps `aws sagemaker search-training-plan-offerings` and sweeps a
configurable set of reservation durations (default: 24 h, 48 h, 72 h,
1 wk, 2 wk, 30 d), since that API filters strictly on duration.

## Why this helper?

`SearchTrainingPlanOfferings` only returns offerings whose duration matches
your `--duration-hours` value exactly — there is no "show me everything"
mode. To see the full picture of what's available right now in an AZ, you
need to call the API once per duration and merge the results. This script
does that and ranks the matches by effective per-instance hourly rate.

## Prerequisites

1. **AWS CLI v2** on `PATH` — see
   [installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
2. **jq** on `PATH`
   - macOS: `brew install jq`
   - Ubuntu/Debian: `sudo apt-get install jq`
   - Amazon Linux / RHEL: `sudo yum install jq`
3. **AWS credentials** configured (any one of):
   - Default profile: `aws configure`
   - Named profile: `aws configure --profile <NAME>` (then pass `--profile <NAME>`)
   - Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`)
   - EC2 instance role / SSO
4. **IAM permission** on the calling principal:
   - `sagemaker:SearchTrainingPlanOfferings`
   - To later purchase a plan: `sagemaker:CreateTrainingPlan`
5. **(Optional) SageMaker reserved-capacity quota** for the requested
   instance type. If `--count` exceeds your quota, the API returns
   `ResourceLimitExceeded` for that duration and the script silently skips
   it.

## Usage

```
./find-ftp.sh --az <AZ> --instance-type <ML_INSTANCE_TYPE> \
              --count <N> --region <REGION> \
              [--start-after <ISO8601_UTC>] [--profile <AWS_PROFILE>] \
              [--target hyperpod-cluster|training-job] \
              [--durations <H1,H2,...>] [--json] [--verbose]
```

Run `./find-ftp.sh --help` for the built-in help.

### Required arguments

| Flag | Description | Example |
|---|---|---|
| `--az` | Availability Zone name | `us-west-2b` |
| `--instance-type` | SageMaker instance type (with `ml.` prefix) | `ml.p5.48xlarge` |
| `--count` | Number of instances to reserve (integer > 0) | `2` |
| `--region` | AWS region | `us-west-2` |

### Optional arguments

| Flag | Description | Default |
|---|---|---|
| `--start-after` | Earliest acceptable start time (ISO-8601 UTC) | now |
| `--profile` | Named AWS CLI profile for credentials | default credential resolution |
| `--target` | Target resource: `hyperpod-cluster` or `training-job` | `hyperpod-cluster` |
| `--durations` | Comma-separated durations (hours) to probe | `24,48,72,168,336,720` |
| `--json` | Emit machine-readable JSON to stdout | off |
| `--verbose` | Print the full AWS error text to stderr when a per-duration call fails | off |
| `-h`, `--help` | Show help and exit | — |

### Supported instance types

Anything `SearchTrainingPlanOfferings` accepts, currently including:

- `ml.p4d.24xlarge`, `ml.p4de.24xlarge`
- `ml.p5.4xlarge`, `ml.p5.48xlarge`
- `ml.p5e.48xlarge`, `ml.p5en.48xlarge`
- `ml.p6-b200.48xlarge`, `ml.p6-b300.48xlarge`
- `ml.p6e-gb200.36xlarge`
- `ml.trn1.32xlarge`, `ml.trn2.48xlarge`

Run `aws sagemaker search-training-plan-offerings help` for the most
current list.

## Examples

### 1. Default target (HyperPod cluster) with default credentials

```bash
./find-ftp.sh \
  --az us-west-2b \
  --instance-type ml.p5.48xlarge \
  --count 2 \
  --region us-west-2 \
  --start-after 2026-06-15T00:00:00Z
```

### 2. Search for SageMaker training-job capacity

```bash
./find-ftp.sh \
  --az us-west-2b \
  --instance-type ml.p5.48xlarge \
  --count 2 \
  --region us-west-2 \
  --start-after 2026-06-15T00:00:00Z \
  --target training-job
```

### 3. Named AWS profile, custom durations

```bash
./find-ftp.sh \
  --az us-west-2b \
  --instance-type ml.p6-b200.48xlarge \
  --count 2 \
  --region us-west-2 \
  --start-after 2026-06-15T00:00:00Z \
  --profile my-aws-profile \
  --durations 24,48,168
```

### 4. Machine-readable JSON

```bash
./find-ftp.sh \
  --az us-west-2b --instance-type ml.p5.48xlarge \
  --count 2 --region us-west-2 --json | jq '.recommendation'
```

JSON shape:

```json
{
  "query": {
    "availability_zone": "us-west-2b",
    "instance_type": "ml.p5.48xlarge",
    "instance_count": 2,
    "region": "us-west-2",
    "start_after": "2026-06-15T00:00:00Z",
    "target_resources": "hyperpod-cluster",
    "durations_hours": [24, 48, 72, 168, 336, 720]
  },
  "count": 9,
  "offerings": [ /* sorted by per_inst_hr ascending */ ],
  "recommendation": { /* the cheapest offering, or null */ }
}
```

### 5. Verbose error output (debugging)

```bash
./find-ftp.sh \
  --az us-west-2b --instance-type ml.p5.48xlarge \
  --count 2 --region us-west-2 --verbose
```

Without `--verbose`, transient or configuration errors per duration appear
as a single `WARNING:` line. With `--verbose`, the full AWS error text is
also printed (to stderr) so you can see exactly what failed.

## Sample output (text mode)

```
Searching SageMaker FTP offerings:
  AZ:             us-west-2b
  Instance type:  ml.p5.48xlarge
  Count:          2
  Region:         us-west-2
  Start after:    2026-06-15T00:00:00Z
  Target:         hyperpod-cluster
  Profile:        <default>
  Durations (h):  24 48 72 168 336 720
  Verbose:        off

Found 9 matching offering(s) in us-west-2b:

OFFERING_ID                                          DUR(h)      UPFRONT  $/inst/hr   START (UTC)          END (UTC)
--------------------------------------------------   ------ ------------ ----------   -------------------- --------------------
tpo-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx..       24      1830.76      38.14   2026-06-15 11:30     2026-06-16 11:30
tpo-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy..       48      3741.12      38.97   2026-06-15 11:30     2026-06-17 11:30
...

Recommendation (lowest $/instance/hour):
  Offering ID:  tpo-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  Duration:     24 h
  AZ:           us-west-2b
  Upfront fee:  1830.76 USD
  Effective:    $38.14 / instance / hour
  Window:       2026-06-15 11:30 UTC -> 2026-06-16 11:30 UTC

To purchase this offering (NON-REFUNDABLE; charges the upfront fee immediately):
  aws sagemaker create-training-plan \
    --region us-west-2 \
    --training-plan-name <YOUR_PLAN_NAME> \
    --training-plan-offering-id tpo-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## How it works

1. For each duration in the sweep list, the script calls:

   ```
   aws sagemaker search-training-plan-offerings \
     --region <REGION> \
     --instance-type <INSTANCE_TYPE> \
     --instance-count <COUNT> \
     --start-time-after <START_AFTER> \
     --duration-hours <DUR> \
     --target-resources hyperpod-cluster
   ```

2. Filters returned offerings to those whose `ReservedCapacityOfferings[].AvailabilityZone` matches `--az`.

3. Computes `$/instance/hour = UpfrontFee / (DurationHours * InstanceCount)`.

4. Prints the matches as a table (text mode) or JSON (`--json`), and surfaces
   the offering with the lowest `$/instance/hour` along with a ready-to-run
   `create-training-plan` command.

The script never purchases anything. To actually reserve the capacity you
must run the printed `create-training-plan` command yourself.

## Error handling per duration

The script calls the API once per duration in the sweep list. Failures are
classified and reported as follows:

| Outcome | Default behavior | With `--verbose` |
|---|---|---|
| Success, has offerings | Process them | Same |
| Success, no offerings | Skip silently | Same |
| `ResourceLimitExceeded` (account quota below requested count) | One-line `Note:` to stderr, then continue | Same + full AWS error text |
| Any other error (expired creds, throttling, invalid region, service outage, etc.) | One-line `WARNING:` to stderr, then continue | Same + full AWS error text |

This means a real configuration problem (e.g. expired credentials) will not
silently masquerade as "no offerings found" — you'll see a `WARNING:` line
explaining what happened. Use `--verbose` if you need the full AWS error
output to debug further.

## Choosing a target resource

Training plans are pinned to a target resource at purchase time and cannot
be reused across targets:

| `--target` value | Use for |
|---|---|
| `hyperpod-cluster` (default) | HyperPod clusters orchestrated by Slurm or EKS. The plan provides compute to a cluster instance group. |
| `training-job` | Standalone SageMaker training jobs scheduled and run via the SageMaker training jobs API. |

Pick the target that matches how you intend to consume the capacity. The
two pools are separate, so an empty result with one target does not imply
the other is also empty — try both if you have flexibility.

## Interpreting empty results

`No offerings found ...` usually means one of the following:

- No capacity is currently being offered in that AZ for those parameters.
- Your account-level reserved-capacity quota for that instance type is
  below `--count`, so the API returned `ResourceLimitExceeded` for every
  duration (silently skipped).
- The `--start-after` window is too narrow.

Things to try:

1. A different AZ in the same region.
2. A smaller `--count`.
3. A later `--start-after`.
4. A different `--region`.
5. The other `--target` (e.g. switch from `hyperpod-cluster` to
   `training-job` or vice versa) — the two capacity pools are separate.
6. Different durations, e.g. `--durations 96,240,480`.
7. Request a quota increase via AWS Service Quotas for the relevant
   `<INSTANCE_TYPE> for cluster usage` quota.

## Important notes

- **Purchase is non-refundable.** `create-training-plan` charges the
  upfront fee immediately and cannot be cancelled. This script intentionally
  only prints the command — it never runs it for you.
- **Single-AZ requirement for EFA workloads.** HyperPod jobs that use EFA
  (e.g. `p4d`, `p5`, `p6-b200`, `trn1`) require all instances to be in the
  same AZ. The AZ you pass here is where the reservation lives.
- **Duration filter is strict.** The API does not return offerings with
  durations other than the value passed via `--duration-hours`. Use
  `--durations` to override the default sweep list if you need a duration
  not covered by the defaults.
- **AZ names vs AZ IDs.** AZ names like `us-west-2b` are scoped to your
  account — the same name maps to different physical AZs in different
  accounts. To compare across accounts, use AZ IDs:
  ```bash
  aws ec2 describe-availability-zones --region us-west-2 \
    --query 'AvailabilityZones[].[ZoneName,ZoneId]' --output table
  ```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `aws CLI not found on PATH` | AWS CLI not installed | Install AWS CLI v2 |
| `jq not found on PATH` | `jq` not installed | `brew install jq` / `apt-get install jq` |
| `WARNING: ... Unable to locate credentials` | No AWS credentials resolved | Configure a profile or set env vars; use `--profile` |
| `WARNING: ... AccessDeniedException` | Missing IAM permission | Grant `sagemaker:SearchTrainingPlanOfferings` |
| `WARNING: ... ThrottlingException` | API rate-limited transiently | Retry; if persistent, run with fewer durations |
| `WARNING: ... Could not connect to the endpoint URL` | Invalid `--region` or network issue | Check the region name; try `--verbose` for details |
| `Note: ... ResourceLimitExceeded` for every duration | Account quota for the instance type is below `--count` | Lower `--count`, or request a quota increase via Service Quotas |
| Result table is empty (no `Note:` or `WARNING:` lines) | No matching capacity in that AZ | Try different AZ, smaller `--count`, later start, different durations |
| Same offering shown multiple times across durations | Different reservation windows / capacity blocks | Expected — each duration is searched independently |

## References

- [SageMaker Training Plans overview](https://docs.aws.amazon.com/sagemaker/latest/dg/reserve-capacity-with-training-plans.html)
- [SearchTrainingPlanOfferings API reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_SearchTrainingPlanOfferings.html)
- [CreateTrainingPlan API reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingPlan.html)
- [SageMaker HyperPod documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html)
