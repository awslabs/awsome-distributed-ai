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
