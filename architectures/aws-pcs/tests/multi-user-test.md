# Test: Multi-User Directory (OpenLDAP)

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
sudo -E bash /usr/local/bin/ldap-add-user.sh testuser1 10001 3000
```

Or manually:
```bash
ldapadd -x -H ldap://localhost -D "cn=admin,dc=cluster,dc=internal" -w "$LDAP_ADMIN_PASSWORD" <<EOF
dn: uid=testuser1,ou=People,dc=cluster,dc=internal
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
uid: testuser1
cn: Test User 1
sn: User1
uidNumber: 10001
gidNumber: 3000
homeDirectory: /home/testuser1
loginShell: /bin/bash
EOF

ldappasswd -x -H ldap://localhost -D "cn=admin,dc=cluster,dc=internal" \
  -w "$LDAP_ADMIN_PASSWORD" -s "testpass123" "uid=testuser1,ou=People,dc=cluster,dc=internal"
```

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
sudo -E bash /usr/local/bin/ldap-add-user.sh testuser2 10002 3000
```

### B6. Delete a user

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

---

## Verdict checklist

| Check | Expected |
|---|---|
| slapd running on login node | ✅ |
| LDAP DB on /home/ldap-db (shared OpenZFS) | ✅ |
| Admin password in SSM Parameter Store | ✅ |
| User created via ldapadd/helper script | ✅ |
| User visible on login node (`getent passwd`) | ✅ |
| User visible on compute node (`srun getent passwd`) | ✅ |
| Home dir auto-created on first login | ✅ |
| User deleted, no longer resolvable | ✅ |
| Slurm job runs as LDAP user | ✅ |
| Multiple nodes resolve same UID | ✅ |
| Home dir accessible from all nodes | ✅ |
| New compute node resolves existing users | ✅ |
| SSSD cache works during brief LDAP outage | ✅ |
| LDAP DB survives login node replacement | ✅ |

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
  --with-decryption --query 'Parameter.Value' --output text --region us-east-2)

sudo LDAP_ADMIN_PASSWORD="$ADMIN_PW" ldap-add-user alice 10001 3000
sudo LDAP_ADMIN_PASSWORD="$ADMIN_PW" ldap-add-user bob 10002 3000
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
sudo LDAP_ADMIN_PASSWORD="$ADMIN_PW" ldap-add-user charlie 10003 3000
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

## Verdict checklist

| Check | Expected |
|---|---|
| sacctmgr show cluster → cluster listed | ✅ |
| Users registered in accounting (alice, bob) | ✅ |
| Jobs run as LDAP users complete successfully | ✅ |
| sacct shows correct user/account/state per job | ✅ |
| sreport shows per-user utilization | ✅ |
| Resource limit enforcement (if set) blocks over-limit jobs | ✅ |
| Unregistered user behavior matches AccountingPolicyEnforcement setting | ✅ |
