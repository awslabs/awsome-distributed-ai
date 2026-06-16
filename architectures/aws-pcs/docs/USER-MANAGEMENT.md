# User Management Guide

This guide covers multi-user management for the PCS reference architecture.
By default, the cluster runs as a single `ubuntu` user (the PCS-Ready DLAMI
default). When `DeployDirectory=true` is set, an OpenLDAP directory is deployed

---

## Access methods (multi-user)

| Method | Pro | Con | Best for |
|---|---|---|---|
| **SSM Session Manager** | No port opening, IAM-based auth, full audit trail in CloudTrail | Each user needs IAM credentials + SSM plugin installed; lands as `ssm-user`, requires `su - <user>` to switch | Admin-only access, security-sensitive environments |
| **SSH over SSM** (`ProxyCommand`) | No port opening + standard SSH workflow (VS Code Remote, scp, rsync) | Each user needs IAM credentials + SSM plugin + SSH config entry; slightly more setup | Teams with IAM already in place, remote IDE use |
| **Direct SSH** (port 22 on login node) | Simplest for users — standard `ssh user@host`, familiar HPC workflow, VS Code/JupyterLab tunnels work natively | Requires SG port 22 open (CIDR-restricted), SSH key distribution needed | Traditional HPC teams, workshop/training environments |

**Recommendation for multi-user clusters**: use **Direct SSH** with CIDR-restricted
port 22 on the login node. This matches the standard HPC user experience and
avoids per-user IAM setup overhead. SSH keys are stored in each user's
`/home/<user>/.ssh/authorized_keys` (shared OpenZFS, visible on all nodes).

### SSH key management

| Approach | Who adds keys | How |
|---|---|---|
| Admin provisions at user-creation time | Cluster admin | Pass `ssh-pub-key` to `ldap-add-user.sh` → written to `/home/<user>/.ssh/authorized_keys` |
| User self-service | Each user | User logs in (first time via admin-provided temp password or SSM) and adds their own key to `~/.ssh/authorized_keys` |
| LDAP-stored keys (`AuthorizedKeysCommand`) | Admin | `sshd_config` queries LDAP for `sshPublicKey` attribute via helper script — centralizes key management but adds sshd config complexity |

For most clusters, the simplest path is: admin creates the user + writes their
SSH public key to `/home/<user>/.ssh/authorized_keys` during creation.

---

## Slurm accounting with PCS

PCS manages the Slurm accounting database internally — you do NOT need to
configure `slurmdbd` or manage a MySQL/MariaDB database. When
`ManagedAccounting=enabled` is set on the cluster, Slurm's accounting commands
(`sacctmgr`, `sacct`, `sreport`) work against PCS's managed backing store.

**What you still need to do manually**:
- `sacctmgr add account <team>` — create Slurm accounts (one per team/project)
- `sacctmgr add user <username> account=<team>` — map LDAP users to Slurm accounts
- (Optional) Set fairshare / QOS via `sacctmgr modify user`

If `AccountingPolicyEnforcement=associations,limits,safe` is set, a user
without a `sacctmgr` entry will have their jobs **rejected**. With the default
`AccountingPolicyEnforcement=none`, unregistered users can still submit jobs
(they just won't have accounting records).

---

## Template structure (how multi-user is wired)

```
pcs-ml-cluster-deploy-all.yaml
│
│  Parameters: DeployDirectory, DirectoryDomainSuffix
│
├─► PrerequisitesStack (ml-cluster-prerequisites.yaml)
│     VPC, FSx Lustre, FSx OpenZFS (/home — shared storage for LDAP DB)
│
├─► ClusterStack (cluster.yaml)
│     PCS Cluster, IAM role, SSM param (Grafana pw)
│
├─► LoginNodeGroupStack (add-cng.yaml)
│     │  MonitoringRole=login, DeployDirectory=$DeployDirectory
│     │
│     └─► UserData:
│           if DeployDirectory=true:
│             curl setup-openldap-server.sh → install slapd
│             DB → /home/ldap-db/ (shared OpenZFS)
│             Admin password → /home/ldap-db/.admin-password
│
├─► OnDemandCNGStack (add-cng.yaml)
│     │  MonitoringRole=compute, DeployDirectory=$DeployDirectory
│     │
│     └─► UserData:
│           if DeployDirectory=true:
│             discover login node IP (ec2:DescribeInstances by tag)
│             curl setup-ldap-client.sh → install SSSD
│             ldap_uri = ldap://<login-node-ip>
│
└─► [Optional] PseriesCNGStack (add-cng-p5/p6-*.yaml)
      Same pattern as OnDemandCNG (SSSD client when DeployDirectory=true)
```

When `DeployDirectory=true` is set, an OpenLDAP directory is deployed
on the login node, and all compute nodes are configured as LDAP clients via
SSSD — giving you centralized POSIX user management across the cluster.

---

## Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Login Node                                                   │
│  ┌──────────────┐    ┌──────────────────────────────────┐    │
│  │  slapd       │    │  SSSD (ldap provider, localhost)  │    │
│  │  (OpenLDAP)  │    └──────────────────────────────────┘    │
│  │  DB: /home/  │                                            │
│  │    ldap-db/  │◄─── LDAP queries from compute nodes        │
│  └──────────────┘                                            │
└──────────────────────────────────────────────────────────────┘
         ▲                          ▲
         │ ldap://login-ip:389      │ ldap://login-ip:389
         │                          │
┌────────┴───────┐        ┌────────┴───────┐
│  Compute Node  │        │  Compute Node  │
│  SSSD (ldap)   │        │  SSSD (ldap)   │
│  NSS + PAM     │        │  NSS + PAM     │
└────────────────┘        └────────────────┘
```

**Key properties:**
- LDAP DB is stored on `/home/ldap-db` (shared OpenZFS) — survives login node
  CNG recreation or instance replacement
- All nodes use SSSD with `cache_credentials=true` — users can still log in
  during brief LDAP server unavailability (e.g. login node reboot)
- UID/GID are explicit POSIX attributes in LDAP (not algorithmic mapping) —
  fully deterministic across all nodes and NFS mounts
- Home directories auto-created on first login via `pam_mkhomedir` at `/home/<username>`
  (shared OpenZFS, visible on all nodes immediately)

---

## Enabling multi-user

### With deploy-all

```bash
aws cloudformation create-stack \
  --stack-name my-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-2b \
    ParameterKey=DeployDirectory,ParameterValue=true \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

### With modular deployment (add-cng.yaml)

Pass `DeployDirectory=true` and `DirectoryDomainSuffix=dc=cluster,dc=internal`
to **both** the login CNG and compute CNG stacks. The login CNG installs the
LDAP server; the compute CNG configures the SSSD client pointing at the login
node.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `DeployDirectory` | `false` | Enable OpenLDAP directory on login node |
| `DirectoryDomainSuffix` | `dc=cluster,dc=internal` | LDAP base DN. Change if you want a different domain (e.g. `dc=myorg,dc=com`) |

---

## Managing users

All user management commands run on the **login node** as root (or via `sudo`).

### Retrieve the LDAP admin password

The admin password is auto-generated at first boot and stored in SSM
Parameter Store:

```bash
CLUSTER_ID=<from stack output>
aws ssm get-parameter --name "/pcs/${CLUSTER_ID}/ldap/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text
```

### Add a user

Use the helper script (pre-installed on the login node at
`/usr/local/bin/ldap-add-user.sh`, or fetch from the repo):

```bash
# Usage: ldap-add-user.sh <username> [uid] [gid] [ssh-pub-key]
sudo LDAP_ADMIN_PASSWORD=$(aws ssm get-parameter \
  --name "/pcs/${CLUSTER_ID}/ldap/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text) \
  /usr/local/bin/ldap-add-user.sh alice 10001 3000
```

Or manually with `ldapadd`:

```bash
ADMIN_PW="<retrieved above>"
BASE_DN="dc=cluster,dc=internal"

ldapadd -x -H ldap://localhost -D "cn=admin,${BASE_DN}" -w "${ADMIN_PW}" <<EOF
dn: uid=alice,ou=People,${BASE_DN}
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
uid: alice
cn: Alice
sn: Smith
uidNumber: 10001
gidNumber: 3000
homeDirectory: /home/alice
loginShell: /bin/bash
EOF

# Set initial password
ldappasswd -x -H ldap://localhost -D "cn=admin,${BASE_DN}" -w "${ADMIN_PW}" \
  -s "initial-password-123" "uid=alice,ou=People,${BASE_DN}"
```

### Verify user exists on all nodes

```bash
# On login node
getent passwd alice

# On a compute node (via srun)
srun -N 1 -n 1 bash -c 'getent passwd alice'
```

Expected output:
```
alice:*:10001:3000:Alice:/home/alice:/bin/bash
```

### Add user to Slurm accounting

```bash
export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH

# Create a Slurm account (one per team/project)
sacctmgr add account researchers Description="ML Researchers"

# Add user to account
sacctmgr add user alice Account=researchers
```

### List users

```bash
# All LDAP users
ldapsearch -x -H ldap://localhost -b "ou=People,dc=cluster,dc=internal" uid
```

### Change a user's password

```bash
ldappasswd -x -H ldap://localhost -D "cn=admin,${BASE_DN}" -w "${ADMIN_PW}" \
  -s "new-password" "uid=alice,ou=People,${BASE_DN}"
```

Or the user can change their own password (if `passwordSelfChange` is enabled):
```bash
ldappasswd -x -H ldap://localhost -D "uid=alice,ou=People,${BASE_DN}" \
  -W -s "new-password" "uid=alice,ou=People,${BASE_DN}"
```

### Remove a user

```bash
ldapdelete -x -H ldap://localhost -D "cn=admin,${BASE_DN}" -w "${ADMIN_PW}" \
  "uid=alice,ou=People,${BASE_DN}"

# Also remove from Slurm
sacctmgr remove user alice
```

### Add a group

```bash
ldapadd -x -H ldap://localhost -D "cn=admin,${BASE_DN}" -w "${ADMIN_PW}" <<EOF
dn: cn=ml-team,ou=Groups,${BASE_DN}
objectClass: posixGroup
cn: ml-team
gidNumber: 3001
memberUid: alice
memberUid: bob
EOF
```

### Add user to an existing group

```bash
ldapmodify -x -H ldap://localhost -D "cn=admin,${BASE_DN}" -w "${ADMIN_PW}" <<EOF
dn: cn=ml-team,ou=Groups,${BASE_DN}
changetype: modify
add: memberUid
memberUid: charlie
EOF
```

---

## Running jobs as a specific user

Once users are in LDAP, Slurm can run jobs as those users:

```bash
# User submits their own job (after logging in as themselves)
srun --partition=cpu1 hostname

# Admin runs a job on behalf of a user
sudo -u alice srun --partition=cpu1 hostname

# Verify UID in the job
srun --partition=cpu1 bash -c 'id; whoami'
```

---

## SSH access for LDAP users

LDAP users can connect to the login node via SSM Session Manager (recommended)
or SSH. For SSH:

1. Add the user's public key to their LDAP entry (if using the
   `openssh-lpk` schema) or to `/home/<username>/.ssh/authorized_keys`
   (auto-created by pam_mkhomedir on shared `/home`).

2. Users connect via SSH-over-SSM:
   ```bash
   # ~/.ssh/config
   Host pcs-login
     HostName <login-instance-id>
     User alice
     ProxyCommand aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters 'portNumber=%p'
   ```

---

## UID/GID conventions

| Range | Purpose |
|---|---|
| 0–999 | System users (do not use) |
| 1000 | `ubuntu` (DLAMI default user) |
| 3000 | `clusterusers` group (default group for new users) |
| 3001+ | Custom groups |
| 10000–59999 | LDAP user UIDs (SSSD `min_id`/`max_id` range) |

Use deterministic UIDs (specify in `ldapadd`) rather than letting LDAP
auto-assign. This ensures consistency across all nodes and NFS mounts.

---

## SSSD caching behavior

| Setting | Value | Meaning |
|---|---|---|
| `cache_credentials` | `true` | Users can authenticate even if LDAP server is temporarily down |
| `enumerate` | `true` | `getent passwd` shows all LDAP users (not just those who have logged in) |
| `entry_cache_timeout` | 5400 (default) | Cached user/group entries are valid for 90 minutes |

To force immediate cache refresh on a node (e.g. after adding a user):
```bash
sudo sss_cache -E
```

---

## Troubleshooting

### User not visible on compute node

```bash
# Check SSSD is running
systemctl status sssd

# Check LDAP connectivity from compute node
ldapsearch -x -H ldap://<login-node-ip> -b "dc=cluster,dc=internal" uid=alice

# Force cache refresh
sudo sss_cache -E
sudo systemctl restart sssd

# Check SSSD logs
journalctl -u sssd -n 50
```

### Login node LDAP server not starting

```bash
# Check slapd status
systemctl status slapd
journalctl -u slapd -n 30

# Check AppArmor (common issue with custom DB path)
aa-status | grep slapd
```

### Home directory not created on first login

```bash
# Verify pam_mkhomedir is configured
grep pam_mkhomedir /etc/pam.d/common-session

# Verify /home is mounted (shared OpenZFS)
mount | grep /home
```

---

## Data persistence

| Data | Location | Survives CNG delete? | Survives stack delete? |
|---|---|---|---|
| LDAP database | `/home/ldap-db/` | ✅ (shared OpenZFS) | ❌ (FSx deleted with prereqs) |
| User home dirs | `/home/<username>/` | ✅ (shared OpenZFS) | ❌ (FSx deleted with prereqs) |
| LDAP admin password | SSM Parameter Store | ✅ (account-level) | ✅ (not part of stack) |

**Backup recommendation**: periodically export the LDAP database:
```bash
slapcat -l /home/ldap-backup/ldap-$(date +%Y%m%d).ldif
```

Restore on a fresh login node:
```bash
systemctl stop slapd
slapadd -l /home/ldap-backup/ldap-YYYYMMDD.ldif
chown -R openldap:openldap /home/ldap-db
systemctl start slapd
```

---

## Upgrading to Simple AD or Managed AD

If your team outgrows the self-hosted OpenLDAP (> 50 users, need HA, need
Kerberos), migrate to AWS Simple AD or Managed Microsoft AD:

1. Export users: `slapcat > users.ldif`
2. Deploy Simple AD (requires 2 AZs — see [ROADMAP](./ROADMAP.md))
3. Import users into AD (convert LDIF to AD format)
4. Update compute node SSSD config: change `id_provider` from `ldap` to `ad`,
   switch to `realm join`
5. Decommission slapd on login node

The Slurm accounting entries (`sacctmgr`) don't need to change — Slurm
resolves users via NSS regardless of the identity backend.
