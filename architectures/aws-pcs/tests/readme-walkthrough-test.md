# README Walkthrough (Test 16)

Deploy from the README's Quick Start and follow the "Accessing the Cluster"
and "Accessing the Grafana dashboards" sections **as written** — copy-paste
each command, no reinterpretation. Every command must return a real value
(no `None`, no `Not found`, no empty tag filter). This catches the kind of
tag / CLI / permission drift where docs "look right" but a first-time user
gets stuck at the connect step.

**When to run:** every PR touching README §3, §6, §8.2, or the tags /
policies those sections rely on.

## What "as written" means

- Only substitute the two variables the README asks the reader to set
  (`STACK_NAME`, `AWS_REGION` where applicable, and the required deploy
  parameter `PrimarySubnetAZ` in §3). Do not swap any other value.
- Prefer running from AWS CloudShell (matches the README's stated
  environment); a local shell with `aws --version >= 2.15` and the
  Session Manager plugin installed is equivalent.

## Steps

### 1. Deploy (README §3 Quick Start)

Run the exact `aws cloudformation create-stack` snippet from §3 with only
`PrimarySubnetAZ` set. Wait for `CREATE_COMPLETE`.

**Gate:** `aws cloudformation describe-stacks --stack-name pcs-ml-cluster
--query 'Stacks[0].StackStatus'` = `CREATE_COMPLETE`.

### 2. Connect to the login node (README §6, CLI variant)

Copy the multi-line `LOGIN_INSTANCE_ID=$(…)` snippet **verbatim** from §6.
Set `STACK_NAME` and `AWS_REGION` at the top.

**Gates:**

- `echo $LOGIN_INSTANCE_ID` prints an `i-…` value (not empty, not `None`)
- `aws ssm start-session --target "$LOGIN_INSTANCE_ID" --region "$AWS_REGION"`
  drops you into a shell on the login node
- Inside, `sudo su - ubuntu && sinfo` lists at least the default `cpu1`
  partition

### 3. Grafana password (README §8.2 "Accessing the Grafana dashboards")

Run the `aws ssm get-parameter` snippet with the stack's `ClusterId`
(fetched from CFN Outputs per the README).

**Gate:** returns a plaintext password (not `ParameterNotFound`).

### 4. Grafana port-forward (README §8.2 Option A)

Run the exact block from §8.2 Option A. It reuses `STACK_NAME` /
`AWS_REGION` from Step 2 and produces `LOGIN_INSTANCE_ID` the same way;
then `aws ssm start-session --document-name AWS-StartPortForwardingSession …`.

**Gates:**

- The port-forward session prints `Port 8443 opened for sessionId …`
- `curl -sk -o /dev/null -w '%{http_code}\n' https://localhost:8443/grafana/login`
  returns `200`
- In a browser, `https://localhost:8443/grafana/` accepts `admin` + the
  password from Step 3 and shows the pre-built dashboards
  (Cluster Summary, Slurm Detail, GPU Node List, …)

### 5. Grafana public IP lookup (README §8.2 Option B, if `GrafanaAccessCidr` was set)

Run the exact block. Skip this step if you did not set `GrafanaAccessCidr`
at deploy time.

**Gate:** returns an IPv4 address (not `None`).

### 6. Cleanup

`aws cloudformation delete-stack --stack-name pcs-ml-cluster`.

**Gate:** `DELETE_COMPLETE`, all nested stacks removed.

## Failure classification

- **Any snippet returns `None` / empty:** a docs snippet has stale tag or
  CFN Output references — do not merge until the snippet is fixed. This
  is the specific regression this test exists to catch.
- **`aws ssm start-session` fails "AccessDenied":** IAM policy scope
  drifted (e.g. `ssm:resourceTag/Name` no longer matches the login node's
  actual Name tag). Fix the policy and the docs in lock-step.
- **Grafana `curl` = 403 / 502 or dashboards missing:** monitoring
  stack didn't come up — see the login node's `/var/log/monitoring-install.log`.
  Not this test's fault, but the walkthrough surfaces it.

## Optional: multi-user variant

If the PR touches multi-user paths (§8.3), redeploy with
`DirectoryService=OpenLDAP-LoginNode` and additionally verify the
`ldap-add-user.sh` + `srun` steps from `docs/USER-MANAGEMENT.md`. Same
"as-written" rule.
