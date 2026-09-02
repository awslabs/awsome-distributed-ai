# User Management Guide

**By default a cluster runs as a single `ubuntu` user** — the PCS-Ready
DLAMI default account, shared across the login and every compute node.
Nothing further is needed for single-operator workflows.

**Read on only if you need multi-user access** (multiple engineers with
their own POSIX accounts, home directories, and Slurm job ownership).
This guide covers the day-to-day operations of a cluster deployed with
`DirectoryService=OpenLDAP-LoginNode`, in which an OpenLDAP directory
runs on the login node and provides centralized POSIX accounts visible
on every node via SSSD. Written for cluster admins who may not be
familiar with LDAP.

---

## 1. Enabling multi-user

Multi-user is a **deploy-time parameter**, not something you can flip
on after the stack exists. Set it via either the AWS console or the
CLI — pick one.

### deploy-all — from the CloudFormation console (Quick Create)

1. Open the deploy-all template's **Launch Stack** link in the
   [README](../README.md#3-quick-start) (or navigate to CloudFormation
   → *Create stack* → *Upload* `pcs-ml-cluster-deploy-all.yaml`).
2. On the *Specify stack details* page, set:
   - **`DirectoryService`** → `OpenLDAP-LoginNode` (default is `none`
     → single-user; **must be changed here** to get multi-user)
   - **`SSHAccessCidr`** → your office CIDR (e.g. `203.0.113.10/32`),
     because LDAP users log in over SSH — see
     [§2.3 Log in as that user](#23-log-in-as-that-user). Leave empty
     to disable direct SSH; users then log in only via SSH-over-SSM.
3. The other parameters (VPC/CIDR, instance types, monitoring, etc.)
   have sensible defaults — see [PARAMETERS.md](./PARAMETERS.md).
4. Acknowledge the two IAM capabilities and *Create stack*.

### deploy-all — from the CLI

```bash
aws cloudformation create-stack \
  --stack-name pcs-ml-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/aws-pcs/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-2b \
    ParameterKey=DirectoryService,ParameterValue=OpenLDAP-LoginNode \
    ParameterKey=SSHAccessCidr,ParameterValue=<your-office-cidr>/32 \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

Both paths produce the same stack — the console *Quick Create* form
just fills in the same `--parameters` for you. Use whichever fits
your workflow.

### modular deployment

Pass to your `add-cng.yaml` stacks:

- Login CNG: `DirectoryRole=server`, `IamProfileArn=<LoginInstanceProfileArn>`
- Compute CNG: `DirectoryRole=client`, `IamProfileArn=<InstanceProfileArn>`

Both share `DirectoryDomainSuffix=dc=cluster,dc=internal` (default).

> **IAM profile matters.** `cluster.yaml` outputs `LoginInstanceProfileArn`
> (login-only, grants SSM write of the OpenLDAP admin secret) and
> `InstanceProfileArn` (compute). Give the login CNG the login profile;
> otherwise OpenLDAP setup fails silently. deploy-all wires this
> automatically.

| Parameter | Default | Purpose |
|---|---|---|
| `DirectoryService` (deploy-all) | `none` | `OpenLDAP-LoginNode` enables the flow |
| `DirectoryRole` (add-cng) | `none` | `server` on the login CNG, `client` on compute CNGs |
| `DirectoryDomainSuffix` | `dc=cluster,dc=internal` | LDAP base DN |

---

## 2. Adding your first LDAP user

Walkthrough for the first LDAP user on a fresh cluster — recover the
admin password, add the user with an SSH key, log them in, and prove
their Slurm jobs actually run on a compute node. Once you've done this
once, subsequent users go through the shorter recipes in
[§3 Detailed operations](#3-detailed-operations).

§2.1–§2.2 run on the login node (SSM session or SSH as `ubuntu`);
§2.3–§2.4 are the LDAP user's own login and job submission from their
workstation.

### 2.1 Recover the LDAP admin password

The source of truth is SSM Parameter Store, populated by the login node
at first boot. Print it once at the start of your session — every raw
`ldap*` command below uses `-W` and every `ldap-add-user.sh` call reads
an env var or prompts, so the value never has to land on the command
line or in shell history.

The login node reads its own cluster ID and region from IMDS, and its
instance role is already granted `ssm:GetParameter` on
`/pcs/<id>/ldap/*`:

```bash
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
CLUSTER_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/tags/instance/aws:pcs:cluster-id)
AWS_REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)

aws ssm get-parameter --region "$AWS_REGION" --with-decryption --name "/pcs/$CLUSTER_ID/ldap/admin-password" --query 'Parameter.Value' --output text
```

> **If SSM has no password** (rare — instance role lacked permission at
> first boot), fall back to the copy on shared storage:
>
> ```bash
> sudo cat /home/ldap-db/.admin-password
> ```
>
> If a later `ldap*` command replies *"Invalid credentials"*, the password
> was typed wrong at the `-W` prompt — re-run the command to re-prompt,
> and re-fetch from SSM here if you've lost it.

### 2.2 Add your first user

The `ldap-add-user.sh` helper takes `<username> <uid> <gid>
[ssh-public-key]`. For the very first user, `10001` is safe — that's the
bottom of the LDAP UID range. For subsequent users pick 10002, 10003,
etc.; the helper refuses a duplicate `uidNumber` or username, so a
collision fails loudly instead of silently sharing a POSIX principal on
the shared `/home` + `/fsx`.

Get the user's SSH public key. If they don't have one yet, they can
generate a keypair on their own workstation:

```bash
# On the user's workstation
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "alice@laptop"
# Then send you the *public* half (~/.ssh/id_ed25519.pub, one line).
```

Back on the login node, add the user. The helper prompts for the admin
password if it isn't already in the environment:

```bash
sudo /usr/local/bin/ldap-add-user.sh alice 10001 3000 "ssh-ed25519 AAAA... alice@laptop"
# LDAP admin password: <paste the value from §2.1 — input is hidden>
```

The script prints the user's initial password. Hand it to the user if
they need to set their own later ([§3.4](#34-reset-a-password));
otherwise they can log in with the SSH key you added and change it
themselves.

Verify:

```bash
getent passwd alice
# alice:*:10001:3000:alice:/home/alice:/bin/bash
```

### 2.3 Log in as that user

Pick one of these two paths depending on how the user reaches AWS. If
`SSHAccessCidr` was set at deploy time, direct SSH is simpler; otherwise
use SSH-over-SSM which needs no inbound port 22.

**Direct SSH** — from the user's workstation, using the private key that
matches the public key above. Get the login node's **public IP** first
(the stock `cluster-user-iam` policy grants the AWS CLI calls used
here):

```bash
STACK_NAME=pcs-ml-cluster                 # your deploy-all CloudFormation stack name
AWS_REGION=us-east-2                      # your region

CLUSTER_ID=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" --query 'Stacks[0].Outputs[?OutputKey==`ClusterId`].OutputValue' --output text)
LOGIN_CNG_ID=$(aws pcs list-compute-node-groups --cluster-identifier "$CLUSTER_ID" --region "$AWS_REGION" --query 'computeNodeGroups[?name==`login`].id' --output text)
aws ec2 describe-instances --region "$AWS_REGION" --filters "Name=tag:aws:pcs:compute-node-group-id,Values=$LOGIN_CNG_ID" "Name=instance-state-name,Values=running" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

Then, from the user's workstation:

```bash
# alice@laptop
ssh -i ~/.ssh/id_ed25519 alice@<login-node-public-ip>
```

> **After a login-node replacement the public IP and SSH host key both
> change.** The login node has no Elastic IP; a replacement gets a fresh
> IP and a new host key, so users see
> `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!` — clear the old
> entry (`ssh-keygen -R <old-ip>`) and reconnect. SSH-over-SSM avoids
> this because it targets the instance ID, not an IP.

**SSH over SSM** — no inbound port 22 needed. Requires the AWS CLI +
the Session Manager plugin on the user's workstation and IAM
credentials scoped to `ssm:StartSession` (the stock `cluster-user-iam`
policy grants exactly that on the login node).

Quick one-liner to verify connectivity without editing any config
files — the login instance ID is resolved from the PCS API, so no
hard-coded `i-0abc…` and no dependency on tag naming:

```bash
STACK_NAME=pcs-ml-cluster                 # your deploy-all CloudFormation stack name
AWS_REGION=us-east-2                      # your region

CLUSTER_ID=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" --query 'Stacks[0].Outputs[?OutputKey==`ClusterId`].OutputValue' --output text)
LOGIN_CNG_ID=$(aws pcs list-compute-node-groups --cluster-identifier "$CLUSTER_ID" --region "$AWS_REGION" --query 'computeNodeGroups[?name==`login`].id' --output text)
LOGIN_INSTANCE_ID=$(aws ec2 describe-instances --region "$AWS_REGION" --filters "Name=tag:aws:pcs:compute-node-group-id,Values=$LOGIN_CNG_ID" "Name=instance-state-name,Values=running" --query 'Reservations[0].Instances[0].InstanceId' --output text)

ssh -i ~/.ssh/id_ed25519 -o ProxyCommand="aws ssm start-session --target $LOGIN_INSTANCE_ID --document-name AWS-StartSSHSession --parameters portNumber=%p --region $AWS_REGION" alice@"$LOGIN_INSTANCE_ID"
```

For day-to-day use, put the instance ID in `~/.ssh/config` so `ssh
pcs-login` just works. The ID changes if the login node is replaced;
re-run the resolver above and update `HostName`.

```
# ~/.ssh/config on the user's workstation
Host pcs-login
  HostName <paste-LOGIN_INSTANCE_ID-here>
  User alice
  IdentityFile ~/.ssh/id_ed25519
  ProxyCommand aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p --region us-east-2
```

Then:

```bash
ssh pcs-login
```

Either way, on first login `pam_mkhomedir` creates `/home/alice` from
`/etc/skel`. The home directory lives on shared OpenZFS and is visible
from every compute node.

### 2.4 Verify on a compute node — as the user

Confirm the end-to-end path: **the user themselves** submits a job and
it runs on a compute node under their own UID / home directory. This
is what a real workflow looks like, so it exercises name resolution,
`$HOME` mounting, and Slurm's numeric-UID launch in one shot.

> **If the cluster was deployed with
> `AccountingPolicyEnforcement=associations,limits,safe`**, this srun
> will fail with *"Invalid account or account/partition combination
> specified"* until you (1) register alice in Slurm accounting —
> jump to [§4 Slurm accounting](#4-slurm-accounting) first — and
> (2) add `--account=<account>` to the srun below. On the default
> (`enforcement=none`) accounting registration is optional and this
> step works as-is with no `--account` argument.

Still logged in as alice from [§2.3](#23-log-in-as-that-user):

```bash
# alice@login
srun -N 1 -n 1 -p cpu1 --time=2:00 bash -c 'id; hostname; touch $HOME/.pcs-verify-ok && ls -l $HOME/.pcs-verify-ok'
# uid=10001(alice) gid=3000(clusterusers) groups=3000(clusterusers)
# cpu1-1
# -rw-rw-r-- 1 alice clusterusers 0 ... /home/alice/.pcs-verify-ok
```

> **`--time=<mm>`** is required on clusters with a `GrpTRESRunMins`
> quota. Without it Slurm estimates the job's TRES-minutes as
> UNLIMITED × 1 CPU and holds it pending with
> `AssocGrpCPURunMinutesLimit`. PCS partitions ship with no
> `DefaultTime`; keep `--time=<mm>` on every submission.

If the user does not resolve immediately, see
[§5 Troubleshooting](#51-user-not-found-on-compute).

---

## 3. Detailed operations

Every raw `ldap*` command uses `-W` to be prompted for the admin
password interactively — nothing sensitive lands on the command line
or in shell history. `ldap-add-user.sh` does the same (env var or
prompt).

### 3.1 Add a user

Same `ldap-add-user.sh` invocation as in [§2.2](#22-add-your-first-user).
Pick the next unused `uidNumber`; [§3.2 List users](#32-list-users) shows
what's already in use. The script refuses a `uidNumber` or username that
already exists — two users sharing a UID would collapse into the same
POSIX principal on the shared `/home` and `/fsx`.

```bash
sudo /usr/local/bin/ldap-add-user.sh <username> <uid> 3000 "<ssh-pub-key>"
# LDAP admin password: <paste from SSM — input is hidden>
```

UID / GID allocation policy for this cluster:

| Range | Purpose |
|---|---|
| 0–999 | System users (do not use) |
| 1000 | `ubuntu` (DLAMI default user) |
| 3000 | `clusterusers` group (default GID for new users) |
| 3001+ | Additional groups |
| 10001–59999 | LDAP user UIDs |

### 3.2 List users

```bash
ldapsearch -x -H ldap://localhost -b "ou=People,dc=cluster,dc=internal" \
  "(objectClass=posixAccount)" uid uidNumber | grep -E '^(uid|uidNumber):'
```

### 3.3 Delete a user

```bash
ldapdelete -x -H ldap://localhost -D "cn=admin,dc=cluster,dc=internal" -W \
  "uid=alice,ou=People,dc=cluster,dc=internal"

# Invalidate the SSSD cache so the deletion is visible immediately.
sudo sss_cache -E
srun -N <compute-nodes> -n <compute-nodes> bash -c 'sudo sss_cache -E'

# Also remove from Slurm accounting if you registered them there.
sudo /opt/aws/pcs/scheduler/slurm-25.11/bin/sacctmgr -i remove user alice

# The user's home directory is NOT deleted automatically.
# Remove it if you want to reclaim the space:
sudo rm -rf /home/alice
```

### 3.4 Reset a password

`-W` prompts for the admin password; `-S` prompts for the new user
password. Neither appears on the command line or in shell history.

```bash
ldappasswd -x -H ldap://localhost -D "cn=admin,dc=cluster,dc=internal" -W -S \
  "uid=alice,ou=People,dc=cluster,dc=internal"
# Enter LDAP Password:              (admin password, hidden)
# New password:                     (new password for alice, hidden)
# Re-enter new password:
```

The user can change their own password after logging in:

```bash
# Run by the user themselves
ldappasswd -x -H ldap://localhost -D "uid=alice,ou=People,dc=cluster,dc=internal" -W -S \
  "uid=alice,ou=People,dc=cluster,dc=internal"
```

### 3.5 Create a group

```bash
ldapadd -x -H ldap://localhost -D "cn=admin,dc=cluster,dc=internal" -W <<EOF
dn: cn=ml-team,ou=Groups,dc=cluster,dc=internal
objectClass: posixGroup
cn: ml-team
gidNumber: 3001
memberUid: alice
memberUid: bob
EOF
```

### 3.6 Add a user to an existing group

```bash
ldapmodify -x -H ldap://localhost -D "cn=admin,dc=cluster,dc=internal" -W <<EOF
dn: cn=ml-team,ou=Groups,dc=cluster,dc=internal
changetype: modify
add: memberUid
memberUid: carol
EOF
```

---

## 4. Slurm accounting

PCS manages the Slurm accounting database internally (enable it with
`ManagedAccounting=enabled` at deploy time). This section walks
through the flow demonstrated in the AWS blog
[Introducing managed accounting for AWS Parallel Computing Service](https://aws.amazon.com/blogs/hpc/introducing-managed-accounting-for-aws-parallel-computing-service/):
register users under project accounts, cap their usage, and read
back utilization. Extra managed-accounting charges (an hourly fee
tied to the controller size + per-GB-month storage governed by the
cluster's Default Purge Time) apply on top of the base cluster —
see the blog for pricing detail.

Registering users in accounting is optional unless the cluster was
deployed with `AccountingPolicyEnforcement=associations,limits,safe`
— with enforcement on, jobs from unregistered users are rejected.
(Slurm also supports a stricter `associations,limits` mode that rejects
over-quota work at submit time rather than holding it pending, but
these templates only offer `none` and `associations,limits,safe`; the
sections below use the latter.)

> **`sacctmgr` add / modify / remove must run as root.** In PCS
> managed accounting the Administrator is `root`; the default
> `ubuntu` user is not an accounting admin, so `sacctmgr -i add …` as
> `ubuntu` fails with *"Only admins/operators/coordinators can add
> accounts"*. Read-only forms (`sacctmgr show …`, `sacct`, `sreport`)
> work as any user.

```bash
S=/opt/aws/pcs/scheduler/slurm-25.11/bin/sacctmgr   # slurm-25.05 for SlurmVersion=25.05
```

### 4.1 Register users to accounts

Reproducing the blog's two-project layout — physics and chemistry
groups sharing one cluster:

```bash
sudo $S -i add account proj_physics   Description="Physics group"
sudo $S -i add account proj_chemistry Description="Chemistry group"

sudo $S -i add user alice Account=proj_physics
sudo $S -i add user bob   Account=proj_physics
sudo $S -i add user carol Account=proj_chemistry

sacctmgr show user alice bob carol WithAssoc \
  format=User,Account,DefaultAccount
```

`WithAssoc` joins the user's association row so per-account
attributes actually appear in the output. Attribute a job to a
project at submit time with `--account=<name>` (see §4.2).

### 4.2 Limits and enforcement

Cap alice at 100 CPU-hours of concurrently-running work
(6000 CPU-minutes):

```bash
sudo $S -i modify user alice set GrpTRESRunMins=cpu=6000

sacctmgr show user alice WithAssoc format=User,Account,GrpTRESRunMins%30
```

With `AccountingPolicyEnforcement=associations,limits,safe` a job
that would put alice over her running-minutes budget is **accepted
at submit time but held pending** with reason
`AssocGrpCPURunMinutesLimit`; it starts only once earlier work drains
the budget. To see this, first tighten the cap so a modest job trips
it (8 CPUs × 30 min = 240 CPU-min > 60):

```bash
sudo $S -i modify user alice set GrpTRESRunMins=cpu=60
```

As alice, in her login shell (`sudo su - alice`) — `sbatch` is on the
login-shell `PATH`, no export needed. `--ntasks-per-node` here must not
exceed what `sinfo -N -o %c` reports for the queue (4 on the default
`cpu1` / `c6i.2xlarge`); at 4 tasks × 30 min = 120 CPU-min > 60 the
limit still trips:

```bash
sudo su - alice
# now inside alice's login shell:
printf '#!/bin/bash\nsleep 1000\n' > ~/myjob.sh && chmod +x ~/myjob.sh
sbatch --account=proj_physics --partition=cpu1 --nodes=1 --ntasks-per-node=4 --time=30:00 ~/myjob.sh
exit
# back in the ubuntu shell:
squeue -u alice -o "%.6i %.10P %.8u %.2t %r"
#   6      cpu1     alice PD AssocGrpCPURunMinutesLimit
```

A job that fits proceeds normally (2 CPUs × 2 min = 4 CPU-min < 60):

```bash
sudo su - alice
printf '#!/bin/bash\nhostname\n' > ~/smalljob.sh && chmod +x ~/smalljob.sh
sbatch --account=proj_physics --partition=cpu1 --nodes=1 --ntasks=1 --time=2:00 ~/smalljob.sh
exit
#   -> Submitted batch job <n> (runs when a node is available)
```

Restore the original cap once done:

```bash
sudo $S -i modify user alice set GrpTRESRunMins=cpu=6000
```

> **`safe` accepts, doesn't `sbatch: error` — even over-quota.** The
> blog's *"Job violates accounting/QOS policy"* message would come from
> Slurm's stricter `associations,limits` mode (no `safe`), which these
> templates don't offer. With `safe` the scheduler still accepts the
> submission so the user can leave work queued; only *starting* it is
> deferred until the running-minutes budget frees up. Watch `squeue`'s
> Reason column, not `sbatch`'s exit status, to see the limit taking
> effect.

### 4.3 Reporting

Same recipes as in the blog:

```bash
# Weekly all-user job listing
sacct --starttime=$(date -d "7 days ago" +%Y-%m-%d) \
  --format="JobID,User,JobName,Partition,Account,AllocCPUS,State,ExitCode"

# Monthly cluster utilization by account/user
sreport cluster AccountUtilizationByUser \
  start=2026-04-01 end=2026-05-01 -t percent \
  format="Accounts,Login,Proper,Used"

# Monthly top users
sreport user topusage start=2026-03-01 end=2026-04-01

# Per-user weekly failure diagnostic
sacct -u alice --starttime=$(date -d "7 days ago" +%Y-%m-%d) \
  --format="JobID,JobName,State,ExitCode,Start,End,MaxRSS,MaxVMSize,Comment"
```

> **Two reporting notes (tool behaviour, not bugs):**
>
> 1. **`sacct --state=…` needs an explicit `-E now`.** A state filter
>    without `-E now` (or `--endtime`) silently returns nothing —
>    `sacct -X --state=FAILED -S 2026-01-01` shows no rows even when
>    failed jobs exist. Always pair it: `sacct -X -a --state=FAILED
>    -S <start> -E now`.
> 2. **`sreport` lags `sacct` by up to an hour.** `sreport` reads
>    slurmdbd's periodic (hourly) usage rollup, so right after jobs
>    finish, `sreport … Used` reads zero until the next rollup boundary.
>    Use `sacct` for up-to-the-second data; use `sreport` for settled
>    historical utilization.

---

## 5. Troubleshooting

### 5.1 "User not found" on compute

> Expected right after a fresh compute node boot. SSSD runs a full
> enumeration in the background on first start; individual lookups can
> briefly return "not found" until it completes (seconds to a minute).
> Jobs still **run** because Slurm launches by numeric UID — this is a
> name-resolution delay, not a job failure.

If it persists:

```bash
srun -N 1 -n 1 bash -c 'systemctl status sssd | head -3'
srun -N 1 -n 1 bash -c 'ldapsearch -x -H ldap://<login-ip> -b dc=cluster,dc=internal uid=alice'
srun -N 1 -n 1 bash -c 'sudo sss_cache -E; sudo systemctl restart sssd'
```

### 5.2 `slapd` not running on the login node

```bash
sudo systemctl status slapd
sudo journalctl -u slapd -n 20
sudo cat /var/log/amazon/pcs/lifecycle/actions/nodeBootstrapped/setup-directory.log
```

### 5.3 Home directory not created

`/home/<user>` is auto-created by `pam_mkhomedir` on first **interactive
login** (SSH or `su -`). Slurm jobs do not create it, so a user who has
never logged in and immediately runs `sbatch` will hit `chdir: No such
file or directory`.

Fix by logging in once as the user, or create it manually:

```bash
sudo mkdir -p /home/alice
sudo chown alice:clusterusers /home/alice
sudo chmod 700 /home/alice
```

### 5.4 New compute node doesn't resolve users

If the node was launched **before** `DirectoryService` was enabled
(e.g. before a stack update), its LaunchTemplate has no SSSD client
setup. Terminate the node; PCS replaces it with a new one that has SSSD.

---

## 6. How it works

- **`slapd` runs on the login node**, DB on shared `/home/ldap-db/`
  (OpenZFS NFS) so it survives login-node restart or replacement.
- **Every node runs SSSD** (server-side on login, client on compute) and
  caches LDAP replies. `sudo sss_cache -E` refreshes.
- **Compute nodes discover the login IP by EC2 tag** at first boot,
  filtering on `pcs-cluster-id=<this cluster>` + `directory-role=server`,
  then set `ldap_uri` in `/etc/sssd/sssd.conf`. Requires
  `ec2:DescribeInstances` on the compute instance role (the cluster
  IAM role grants it).
- **Home directories are on shared `/home`**, auto-created by
  `pam_mkhomedir` at first interactive login.
- **Slurm launches jobs by numeric UID**, so a job runs even if a
  compute node can't currently name-resolve the user (SSSD cold cache,
  brief LDAP outage).

> ⚠️ **Single login node only.** `OpenLDAP-LoginNode` runs slapd on one
> node; keep the login CNG at `MinCount=MaxCount=1` while the directory
> is enabled. Two login nodes would open the same MDB from two
> processes (corruption risk). If HA is needed, plan for AWS Simple AD /
> Managed AD — see [ROADMAP.md](./ROADMAP.md).

**After a login-node replacement**, the new instance re-tags itself
`directory-role=server` and re-attaches the same MDB. Newly-booting
compute nodes discover the new IP; already-running compute nodes still
hold the old `ldap_uri` and need a one-shot fix:

```bash
srun -N <n> -n <n> bash -c 'sudo sed -i "s#ldap_uri = .*#ldap_uri = ldap://<new-login-ip>#" /etc/sssd/sssd.conf && sudo sss_cache -E && sudo systemctl restart sssd'
```

---

## 7. Data persistence and backup

| Data | Location | Survives node replacement? | Survives stack delete? |
|---|---|---|---|
| LDAP database | `/home/ldap-db/` (OpenZFS) | ✅ | ❌ (FSx deleted) |
| Home directories | `/home/<user>/` (OpenZFS) | ✅ | ❌ (FSx deleted) |
| Admin password | SSM Parameter Store | ✅ | ✅ |

### Back up

```bash
# Periodically (e.g. from cron)
sudo slapcat -l "/home/ldap-backup-$(date +%Y%m%d).ldif"
```

### Restore on a fresh login node

```bash
sudo systemctl stop slapd
sudo slapadd -l /home/ldap-backup-YYYYMMDD.ldif
sudo chown -R openldap:openldap /home/ldap-db
sudo systemctl start slapd
```
