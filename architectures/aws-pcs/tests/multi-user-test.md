# Tests 11-12: Multi-User Directory (OpenLDAP) + Slurm Accounting

Run this test when `DirectoryService=OpenLDAP-LoginNode` is enabled. Validates that the
OpenLDAP directory on the login node is functional, that LDAP users are
visible on compute nodes via SSSD, and that Slurm can run jobs as those users.

**Prerequisites**: a deployed cluster with `DirectoryService=OpenLDAP-LoginNode` on the login
CNG and at least one compute CNG. All commands run from the login node unless
noted otherwise.

---

## Part A — LDAP server health (login node)

### A1. slapd service running

```bash
systemctl status slapd
```

Expected: `Active: active (running)`

### A2. LDAP database on shared storage

```bash
ls -la /home/ldap-db/
```

Expected: MDB data files (`data.mdb`, `lock.mdb`) owned by `openldap:openldap`.
This location (shared OpenZFS `/home`) means the DB survives login node
replacement.

### A3. Base DN searchable

```bash
ldapsearch -x -H ldap://localhost -b "dc=cluster,dc=internal" -s base
```

Expected: returns the base entry without errors.

### A4. Organizational units exist

```bash
ldapsearch -x -H ldap://localhost -b "dc=cluster,dc=internal" "(objectClass=organizationalUnit)" dn
```

Expected: `ou=People,dc=cluster,dc=internal` and `ou=Groups,dc=cluster,dc=internal`

### A5. Admin password in SSM

```bash
CLUSTER_ID=<from stack output>
aws ssm get-parameter --name "/pcs/${CLUSTER_ID}/ldap/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text
```

Expected: returns a non-empty string (the auto-generated password).

---

## Part B — User lifecycle

### B1. Create a test user

```bash
export LDAP_ADMIN_PASSWORD=$(aws ssm get-parameter \
  --name "/pcs/${CLUSTER_ID}/ldap/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text)
export LDAP_DOMAIN_SUFFIX="dc=cluster,dc=internal"

# Using the helper script
sudo -E ldap-add-user.sh testuser1 10001 3000
```

(For the manual `ldapadd`/`ldappasswd` equivalent, see
[USER-MANAGEMENT.md](../docs/USER-MANAGEMENT.md).)

### B2. Verify user visible on login node

```bash
getent passwd testuser1
id testuser1
```

Expected:
```
testuser1:*:10001:3000:Test User 1:/home/testuser1:/bin/bash
uid=10001(testuser1) gid=3000(clusterusers) groups=3000(clusterusers)
```

### B3. Verify user visible on compute nodes

```bash
export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH
srun -N 1 -n 1 -p cpu1 bash -c 'getent passwd testuser1; id testuser1'
```

Expected: same output as B2. If the user is not visible, SSSD cache may need
a refresh:
```bash
srun -N 1 -n 1 -p cpu1 bash -c 'sudo sss_cache -E; sleep 2; getent passwd testuser1'
```

### B4. Home directory auto-creation

```bash
# Login as testuser1 (triggers pam_mkhomedir)
sudo su - testuser1 -c 'pwd; ls -la ~'
```

Expected: `/home/testuser1` exists, owned by `testuser1:3000`.

Verify visible from compute:
```bash
srun -N 1 -n 1 -p cpu1 bash -c 'ls -la /home/testuser1'
```

### B5. Create a second user

```bash
sudo -E ldap-add-user.sh testuser2 10002 3000
```

### B5a. Duplicate UID / username is rejected

The helper pre-checks the directory and refuses to add a user that
would share a `uidNumber` with an existing entry (LDAP itself would
happily allow this — two POSIX users with the same UID become the
same principal on the shared `/home` and `/fsx`). Same check for the
username. Regression guard for `ldap-add-user.sh`.

```bash
# Duplicate uidNumber
sudo -E ldap-add-user.sh testuser3 10001 3000; echo "EXIT=$?"
# Expected: EXIT=1, stderr: "uidNumber 10001 is already used by 'testuser1'. Pick a different uid."

# Duplicate username
sudo -E ldap-add-user.sh testuser1 10099 3000; echo "EXIT=$?"
# Expected: EXIT=1, stderr: "Username 'testuser1' already exists in the directory."

# Confirm neither survived
ldapsearch -x -H ldap://localhost -b "ou=People,dc=cluster,dc=internal" \
  "(|(uid=testuser3)(uidNumber=10099))" uid uidNumber
# Expected: no matching entries.
```

### B6. Reset a password (`-W -S` prompt path)

```bash
ldappasswd -x -H ldap://localhost -D "cn=admin,dc=cluster,dc=internal" -W -S \
  "uid=testuser1,ou=People,dc=cluster,dc=internal"
# Enter LDAP Password:              (admin password, hidden)
# New password:                     (new password for testuser1, hidden)
# Re-enter new password:
```

Then verify the user can bind with the new password (still on the login
node):
```bash
ldapwhoami -x -H ldap://localhost -D "uid=testuser1,ou=People,dc=cluster,dc=internal" -W
# Expected: dn:uid=testuser1,ou=People,dc=cluster,dc=internal
```

### B7. Group create and member add (`-W` prompt path)

```bash
# Create group with testuser1 as an initial member
cat > /tmp/g.ldif <<EOF
dn: cn=ml-team,ou=Groups,dc=cluster,dc=internal
objectClass: posixGroup
cn: ml-team
gidNumber: 3001
memberUid: testuser1
EOF
ldapadd -x -H ldap://localhost -D "cn=admin,dc=cluster,dc=internal" -W -f /tmp/g.ldif

sudo sss_cache -E && sleep 2
getent group ml-team
# Expected: ml-team:*:3001:testuser1

# Add testuser2 (still present at this point — B9 deletes it later) to the group
cat > /tmp/m.ldif <<EOF
dn: cn=ml-team,ou=Groups,dc=cluster,dc=internal
changetype: modify
add: memberUid
memberUid: testuser2
EOF
ldapmodify -x -H ldap://localhost -D "cn=admin,dc=cluster,dc=internal" -W -f /tmp/m.ldif

sudo sss_cache -E && sleep 2
id testuser2
# Expected: uid=10002(testuser2) gid=3000(clusterusers) groups=3000(clusterusers),3001(ml-team)

rm -f /tmp/g.ldif /tmp/m.ldif
```

### B8. SSH-over-SSM login as LDAP user (end-to-end user flow)

End-to-end path from an operator laptop that has AWS CLI + IAM permissions
but no direct SSH access to the VPC.

**1. On the laptop — generate a keypair:**

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/pcs-testuser1 -C "testuser1@laptop"
cat ~/.ssh/pcs-testuser1.pub
```

**2. On the login node — install the public key for testuser1.** B1
already created testuser1 without a key, so append the pubkey to the
existing shared-home `authorized_keys` (this is where the helper would
put it too — it does not store keys in LDAP):

```bash
PUBKEY='<paste the ssh-ed25519 line from step 1>'
sudo install -d -m 700 -o testuser1 -g 3000 /home/testuser1/.ssh
echo "$PUBKEY" | sudo tee -a /home/testuser1/.ssh/authorized_keys >/dev/null
sudo chown testuser1:3000 /home/testuser1/.ssh/authorized_keys
sudo chmod 600 /home/testuser1/.ssh/authorized_keys
```

**3. Back on the laptop — resolve the login instance ID and connect over
SSM.** Uses the PCS API (no reliance on the mutable `Name` tag):

```bash
STACK_NAME=<cluster-stack>
AWS_REGION=<region>
LOGIN_INSTANCE_ID=$(aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=tag:aws:pcs:compute-node-group-id,Values=$(aws pcs list-compute-node-groups --cluster-identifier $(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" --query 'Stacks[0].Outputs[?OutputKey==`ClusterId`].OutputValue' --output text) --region "$AWS_REGION" --query 'computeNodeGroups[?name==`login`].id' --output text)" \
              "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

ssh -i ~/.ssh/pcs-testuser1 \
    -o ProxyCommand="aws ssm start-session --target $LOGIN_INSTANCE_ID --document-name AWS-StartSSHSession --parameters portNumber=%p --region $AWS_REGION" \
    testuser1@"$LOGIN_INSTANCE_ID" \
    "id; hostname"
# Expected: uid=10001(testuser1) ..., login-node hostname.

# Submit a job and confirm the shared /home write:
ssh -tt -i ~/.ssh/pcs-testuser1 \
    -o ProxyCommand="aws ssm start-session --target $LOGIN_INSTANCE_ID --document-name AWS-StartSSHSession --parameters portNumber=%p --region $AWS_REGION" \
    testuser1@"$LOGIN_INSTANCE_ID" \
    "bash -lc 'srun -N 1 -n 1 -p cpu1 bash -c \"id; hostname; touch \\\$HOME/.pcs-verify-ok\"; ls -l \$HOME/.pcs-verify-ok'"
# Expected: compute reports uid=10001, and .pcs-verify-ok is visible back on the
# login home (shared OpenZFS).
```

### B9. Delete a user

```bash
ldapdelete -x -H ldap://localhost -D "cn=admin,dc=cluster,dc=internal" \
  -w "$LDAP_ADMIN_PASSWORD" "uid=testuser2,ou=People,dc=cluster,dc=internal"

# Verify deleted
getent passwd testuser2    # should return nothing
```

Verify on compute (after cache expires or forced refresh):
```bash
srun -N 1 -n 1 -p cpu1 bash -c 'sudo sss_cache -E; sleep 2; getent passwd testuser2 || echo "user not found (correct)"'
```

---

## Part C — Slurm integration

### C1. Add user to Slurm accounting

```bash
export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH
sacctmgr -i add account testaccount Description="Test Account"
sacctmgr -i add user testuser1 Account=testaccount
sacctmgr show user testuser1
```

Expected: user listed with account `testaccount`.

### C2. Run a job as LDAP user

```bash
sudo su - testuser1 -c 'export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH; srun -p cpu1 -N 1 -n 1 bash -c "whoami; id; hostname"'
```

Expected:
```
testuser1
uid=10001(testuser1) gid=3000(clusterusers) groups=3000(clusterusers)
cpu1-1
```

### C3. Submit a batch job as LDAP user

```bash
sudo su - testuser1 -c 'export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH; sbatch --wrap="id; hostname" -p cpu1 -o /home/testuser1/test-job.out'
sleep 30
cat /home/testuser1/test-job.out
```

### C4. Verify job ownership in sacct

```bash
sacct -u testuser1 --format=JobID,User,Account,State,ExitCode
```

Expected: job(s) listed under user `testuser1`, account `testaccount`.

---

## Part D — Multi-node consistency

### D1. Multiple nodes resolve the same UID

```bash
srun -N 2 -n 2 -p cpu1 bash -c 'echo "$(hostname): $(id testuser1)"'
```

Expected: both nodes report `uid=10001(testuser1)` — confirms UID consistency
across nodes via shared LDAP.

### D2. Home directory accessible from all nodes

```bash
# Write from one node, read from another
srun -N 1 -n 1 -p cpu1 bash -c 'echo "hello from $(hostname)" > /home/testuser1/multinode-test.txt'
srun -N 1 -n 1 -p cpu1 --exclude=$(srun -N 1 -n 1 -p cpu1 hostname) bash -c 'cat /home/testuser1/multinode-test.txt'
```

Expected: the second node reads what the first wrote (shared OpenZFS `/home`).

### D3. New compute node picks up existing users

If a new compute node scales up after users were created:
```bash
# Force a new node to spin up
srun -N 2 -n 2 -p cpu1 bash -c 'getent passwd testuser1'
```

Expected: the freshly-started node resolves `testuser1` via SSSD (it queries
the login node's LDAP at boot via the client setup script).

---

## Part E — LDAP server resilience

### E1. SSSD cache survives brief LDAP outage

```bash
# Stop slapd briefly on login node
sudo systemctl stop slapd

# Compute node should still resolve from cache
srun -N 1 -n 1 -p cpu1 bash -c 'getent passwd testuser1'

# Restart
sudo systemctl start slapd
```

Expected: cached user resolves even with slapd down (SSSD `cache_credentials=true`).

### E2. LDAP DB survives login node replacement

After login node terminate + PCS replacement:
```bash
# On new login node
ls /home/ldap-db/data.mdb
ldapsearch -x -H ldap://localhost -b "ou=People,dc=cluster,dc=internal" uid
```

Expected: previously-created users still in the directory (DB on shared OpenZFS
survived the instance replacement).

### E3. Admin password is preserved across login node replacement

```bash
# Before terminating the login node, record the hash:
aws ssm get-parameter --name "/pcs/<cluster-id>/ldap/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text | sha256sum
# Terminate the login node, wait for the PCS replacement, then on the new node:
aws ssm get-parameter --name "/pcs/<cluster-id>/ldap/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text | sha256sum
# Admin bind on the new login node:
ldapsearch -x -H ldap://localhost -D "cn=admin,dc=cluster,dc=internal" \
  -w "<password>" -b dc=cluster,dc=internal -s base dn
```

Expected: the SHA-256 is **identical** before and after (the new login node
reuses the existing SSM password instead of regenerating it), and the admin bind
returns `result: 0 Success`.

### E4. Already-running compute after replacement (stale ldap_uri) + recovery

This is the worst case: a compute node is running a job when the login node is
replaced, so it keeps the old login IP cached in `ldap_uri`.

```bash
# 1. Submit a long job so a compute node stays up, confirm it's RUNNING.
# 2. Terminate the login node; wait for the PCS replacement (new private IP).
# 3. Add a NEW user on the new login node, then on the still-running compute:
srun -w <that-node> bash -c 'getent passwd <new-user>'      # -> NOT found (stale ldap_uri)
srun -w <that-node> bash -c 'getent passwd <cached-user>'   # -> still resolves (SSSD cache)
# 4. Recovery: point the node's SSSD at the new login IP and refresh.
srun -w <that-node> bash -c \
  'sudo sed -i "s#ldap_uri = .*#ldap_uri = ldap://<new-login-ip>#" /etc/sssd/sssd.conf \
   && sudo sss_cache -E && sudo systemctl restart sssd'
srun -w <that-node> bash -c 'getent passwd <new-user>'      # -> now resolves
```

Expected: before recovery the new user is unresolvable on the stale node while
cached users still resolve; after the recovery command the new user resolves.
The job that was running throughout is **unaffected** (recovery is non-disruptive;
no node drain).

### E5. A job runs even when its user is unresolvable on the compute node

Confirms the design decision not to drain on a directory gap: Slurm launches by
numeric UID.

```bash
# On a compute node where SSSD can't currently resolve <user> (e.g. mid-recovery
# in E4, or sssd stopped): a job submitted as that user still runs.
sacct -X -a -j <jobid> --format=JobID,User,State,ExitCode -S today -E now
```

Expected: the job reaches `COMPLETED` (ExitCode `0:0`) even though
`id <user>` / `getent passwd <user>` returns "no such user" on that node —
name resolution is degraded, the job is not.


---

# Test: Slurm Managed Accounting + Multi-User

Validates that Slurm managed accounting works with LDAP multi-user setup.
Covers user/account creation, resource limit enforcement, job tracking, and
reporting.

**Prerequisites**:
- Cluster with `ManagedAccounting=enabled` and `DirectoryService=OpenLDAP-LoginNode`
- Slurm 25.11 (accounting is available on 24.11+; templates default to 25.11)
- At least one compute node available

All commands run on the login node as root unless noted.

---

## Part A — Accounting infrastructure

### A1. Verify accounting is active

```bash
export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH
sacctmgr show cluster
```

Expected: cluster name listed with a valid `ControlHost`.

### A2. Default account exists

```bash
sacctmgr show account
```

If no accounts exist yet, create the default:
```bash
sacctmgr -i add account default Description="Default account"
```

---

## Part B — User + account registration

### B1. Create LDAP users (if not done already)

```bash
ADMIN_PW=$(aws ssm get-parameter --name "/pcs/${CLUSTER_ID}/ldap/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text --region <region>)

sudo LDAP_ADMIN_PASSWORD="$ADMIN_PW" ldap-add-user.sh alice 10001 3000
sudo LDAP_ADMIN_PASSWORD="$ADMIN_PW" ldap-add-user.sh bob 10002 3000
```

### B2. Register users in Slurm accounting

```bash
sacctmgr -i add account ml-team Description="ML Research Team"
sacctmgr -i add user alice Account=ml-team
sacctmgr -i add user bob Account=ml-team

# Verify
sacctmgr show user alice bob format=User,Account,DefaultAccount
```

Expected:
```
     User    Account DefaultAccount
--------- ---------- --------------
    alice    ml-team        ml-team
      bob    ml-team        ml-team
```

### B3. Set resource limits (optional)

```bash
# Cap alice at 100 CPU-hours
sacctmgr -i modify user alice set GrpTRESRunMins=cpu=6000

# Verify
sacctmgr show user alice format=User,Account,GrpTRESRunMins
```

---

## Part C — Job submission and tracking

### C1. Submit jobs as LDAP users

```bash
# As alice
sudo su - alice -c 'export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH; \
  sbatch --partition=cpu1 --wrap="sleep 10; hostname; id" -J alice-test'

# As bob
sudo su - bob -c 'export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH; \
  sbatch --partition=cpu1 --wrap="sleep 10; hostname; id" -J bob-test'
```

Wait for jobs to complete:
```bash
squeue  # should show jobs running then empty
```

### C2. Verify job history with sacct

```bash
sacct --starttime=$(date -d "1 hour ago" +%Y-%m-%dT%H:%M) \
  --format="JobID,User,JobName,Partition,Account,AllocCPUS,State,ExitCode,Elapsed"
```

Expected: both jobs listed with correct User, Account, State=COMPLETED.

### C3. Per-user reporting

```bash
# Jobs by alice
sacct -u alice --format="JobID,JobName,State,Start,End,Elapsed"

# Utilization by account
sreport cluster AccountUtilizationByUser \
  start=$(date -d "1 hour ago" +%Y-%m-%dT%H:%M) \
  format="Account,Login,Used"
```

Expected: `sacct -u alice` lists alice's jobs; `sreport` shows alice and bob
under `ml-team` with non-zero `Used`.

### C4. Resource limit enforcement (if B3 was set)

```bash
# Submit a job that would exceed alice's limit
sudo su - alice -c 'export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH; \
  sbatch --partition=cpu1 --ntasks=96 --time=120:00 --wrap="sleep 7200" -J limit-test'
```

Check if job is pending with reason `AssocGrpCPURunMinutesLimit`:
```bash
squeue -u alice --format="%i %j %T %r"
```

---

## Part D — Fairshare (optional)

### D1. Set different fairshare weights

```bash
sacctmgr -i modify account ml-team set FairShare=100
sacctmgr -i add account ops-team Description="Operations"
sacctmgr -i modify account ops-team set FairShare=50
```

### D2. Check fairshare values

```bash
sshare -a --format=Account,User,RawShares,NormShares,RawUsage,FairShare
```

---

## Part E — AccountingPolicyEnforcement

### E1. Test with enforcement=none (default)

An unregistered user can still submit jobs:
```bash
# Create a user NOT in sacctmgr
sudo LDAP_ADMIN_PASSWORD="$ADMIN_PW" ldap-add-user.sh charlie 10003 3000
sudo su - charlie -c 'export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH; \
  srun -p cpu1 -N1 -n1 hostname'
```

Expected: job runs (no accounting enforcement).

### E2. Test with enforcement=associations,limits,safe

If `AccountingPolicyEnforcement=associations,limits,safe` was set at cluster
creation:
```bash
# charlie (not in sacctmgr) should be rejected
sudo su - charlie -c 'export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH; \
  srun -p cpu1 -N1 -n1 hostname'
```

Expected: `error: Unable to allocate resources: Invalid account or account/partition combination specified`

---

## Part F — Blog scenario end-to-end

Reproduces [Introducing managed accounting for AWS Parallel Computing
Service](https://aws.amazon.com/blogs/hpc/introducing-managed-accounting-for-aws-parallel-computing-service/)
on top of the existing LDAP users, exercising the customer-facing flow
the blog demonstrates: two project accounts, a per-user cap, `--account=`
job attribution, quota-based rejection, and the blog's own reporting
commands. Regression guard for the walkthrough in
[USER-MANAGEMENT.md §4.1](../docs/USER-MANAGEMENT.md#41-register-users-to-accounts).

**Prerequisites**: same as the accounting test (`ManagedAccounting=enabled`
+ `DirectoryService=OpenLDAP-LoginNode`); alice / bob / carol as LDAP
users (add carol here if only alice/bob exist).

### F1. Project accounts and quota

```bash
S=/opt/aws/pcs/scheduler/slurm-25.11/bin/sacctmgr   # slurm-25.05 for SlurmVersion=25.05

sudo LDAP_ADMIN_PASSWORD="$ADMIN_PW" ldap-add-user.sh carol 10004 3000

sudo $S -i add account proj_physics   Description="Physics group"
sudo $S -i add account proj_chemistry Description="Chemistry group"
sudo $S -i add user alice Account=proj_physics
sudo $S -i add user bob   Account=proj_physics
sudo $S -i add user carol Account=proj_chemistry

sudo $S -i modify user alice set GrpTRESRunMins=cpu=6000

sacctmgr show user alice bob carol WithAssoc format=User,Account,DefaultAccount,GrpTRESRunMins
```

Expected: three users listed under the two project accounts; alice's
`GrpTRESRunMins` column shows `cpu=6000`.

### F2. Blog-style job submission with `--account=`

Run as alice in her login shell (`sudo su - alice`; exit back to the
`ubuntu` shell when done):

```bash
sudo su - alice
printf '#!/bin/bash\nhostname\nid\n' > ~/smalljob.sh && chmod +x ~/smalljob.sh
sbatch --account=proj_physics --partition=cpu1 --cpus-per-task=2 --time=2:00 ~/smalljob.sh
exit
```

Expected: `Submitted batch job <n>`; `sacct -j <n> -X --format=JobID,User,Account,State` reports `alice / proj_physics / COMPLETED`.

### F3. Quota hold with `safe` enforcement

Only meaningful with `AccountingPolicyEnforcement=associations,limits,safe`
(the enforcement mode these templates offer). Note the `safe` variant
*accepts* the submission and holds it pending instead of returning
`sbatch: error …`.

Tighten alice's limit first so a modest job triggers it (reusing `$S`
from F1):

```bash
sudo $S -i modify user alice set GrpTRESRunMins=cpu=60
```

Submit an over-quota job (4 CPUs × 30 min = 120 CPU-min > 60). Slurm
counts CPUs from what `sinfo -N -o %c` reports for the queue's nodes
(not the EC2 vCPU count), so cap `--ntasks-per-node` at that value —
the default `cpu1` on `c6i.2xlarge` reports 4:

```bash
sudo su - alice
sbatch --account=proj_physics --partition=cpu1 --nodes=1 --ntasks-per-node=4 --time=30:00 --wrap='sleep 1000'
exit
squeue -u alice -o "%.6i %.10P %.8u %.2t %r"
```

Expected: `sbatch: Submitted batch job N`, then `squeue` shows the job
in state `PD` with Reason `AssocGrpCPURunMinutesLimit`. Restore the
original limit when done: `sudo $S -i modify user alice set GrpTRESRunMins=cpu=6000`.

### F4. Blog reporting recipes

```bash
sacct --starttime=$(date -d "7 days ago" +%Y-%m-%d) \
  --format="JobID,User,JobName,Partition,Account,AllocCPUS,State,ExitCode"

sreport cluster AccountUtilizationByUser \
  start=$(date -d "30 days ago" +%Y-%m-%d) end=now \
  -t percent format="Accounts,Login,Proper,Used"

sreport user topusage start=$(date -d "30 days ago" +%Y-%m-%d) end=now

sacct -u alice --starttime=$(date -d "7 days ago" +%Y-%m-%d) \
  --format="JobID,JobName,State,ExitCode,Start,End,MaxRSS,MaxVMSize,Comment"
```

Expected: `sacct` returns the jobs submitted above; `sreport` may print
zero `Used` values for freshly-completed jobs — that's the hourly
slurmdbd rollup lag, not a failure (see USER-MANAGEMENT §4 note #2).

### F5. Cleanup

```bash
sudo $S -i remove user       where User=alice   Account=proj_physics
sudo $S -i remove user       where User=bob     Account=proj_physics
sudo $S -i remove user       where User=carol   Account=proj_chemistry
sudo $S -i remove account    where Account=proj_physics
sudo $S -i remove account    where Account=proj_chemistry
```
