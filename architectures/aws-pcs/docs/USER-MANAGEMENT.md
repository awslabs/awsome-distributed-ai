# User Management Guide

This guide is written for **cluster administrators who may not be familiar with
LDAP**. It covers the day-to-day operations of managing users on a PCS
reference architecture cluster with `DirectoryService=OpenLDAP-LoginNode`.

By default, the cluster runs as a single `ubuntu` user. When multi-user is
enabled, an OpenLDAP directory runs on the login node and provides centralized
POSIX user accounts visible on all nodes via SSSD.

---

## Quick reference (common tasks)

| Task | Command (run on login node as root) |
|---|---|
| Add a user | `sudo ldap-add-user alice 10001 3000` |
| List all users | `sudo ldap-list-users` |
| Delete a user | `sudo ldap-delete-user alice` |
| Reset a user's password | `sudo ldap-reset-password alice` |
| Add user to Slurm accounting | `sacctmgr -i add user alice Account=default` |
| Verify user on compute node | `srun -N1 -n1 -p cpu1 id alice` |

The `ldap-*` commands above are helper scripts installed on the login node at
`/usr/local/bin/`. They wrap raw `ldapadd`/`ldapdelete`/`ldappasswd` commands
so you don't need to know LDAP syntax. The full manual commands are documented
below for reference.

---

## How it works (overview)

```
┌─────────────────────────────────────────────────────────┐
│  Login Node                                              │
│                                                          │
│  ┌──────────┐     ┌──────────┐     ┌──────────────┐    │
│  │  slapd   │────►│  SSSD    │────►│  NSS / PAM   │    │
│  │ (OpenLDAP│     │  (cache) │     │  (getent,    │    │
│  │  server) │     │          │     │   login, su) │    │
│  └──────────┘     └──────────┘     └──────────────┘    │
│       │                                                  │
│  DB: /home/ldap-db/ (shared OpenZFS)                    │
└───────┼──────────────────────────────────────────────────┘
        │ ldap://login-ip:389
        ▼
┌─────────────────────────────────────────────────────────┐
│  Compute Node                                            │
│                                                          │
│  ┌──────────┐     ┌──────────────┐                      │
│  │  SSSD    │────►│  NSS / PAM   │                      │
│  │  (client)│     │  (getent,    │                      │
│  │          │     │   srun user) │                      │
│  └──────────┘     └──────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

**Key points:**
- Users are stored in the LDAP database on the login node
- The database lives on shared `/home` (OpenZFS NFS) — it survives login node restart/replacement
- Every node (login + compute) runs SSSD which queries LDAP for user info
- When you add a user in LDAP, they become visible on all nodes within seconds
- Home directories are auto-created at first login (shared `/home` on OpenZFS)
- Slurm sees LDAP users transparently — no Slurm configuration needed for user resolution

> ⚠️ **Single login node only.** `OpenLDAP-LoginNode` runs the directory server
> on **one** login node, so keep the login node group at `MinCount=MaxCount=1`
> while the directory is enabled. Compute clients discover the server by its
> `directory-role=server` tag and the slapd database is a single MDB on shared
> `/home`; running two login nodes would give clients an ambiguous server and
> have two `slapd` processes open the same database files concurrently
> (corruption risk). If you need multiple login nodes or a highly-available
> directory, use a managed backend (the planned `SimpleAD` / `ManagedAD`
> `DirectoryService` options) rather than the login-node OpenLDAP.

### How a compute node finds the LDAP server (tag-based discovery)

This part is **not obvious**, so it's worth spelling out. A compute node does
**not** receive the login node's IP as a parameter — PCS launches the login and
compute node groups independently, and the login node's private IP isn't known
at template-synthesis time (and it changes if the login node is replaced).
Instead, discovery happens **at compute-node boot**, by EC2 tag lookup:

1. When the directory is enabled, the login node group tags its instance
   `directory-role=server` (alongside `pcs-cluster-id=<this cluster>`). The
   compute node groups tag themselves `directory-role=client`. This
   `directory-role` tag is **dedicated to the directory feature** — it is
   deliberately *separate* from the monitoring stack's `monitoring-role` tag, so
   the two features don't depend on each other.
2. On first boot, each compute node runs `setup-directory.sh client`, which
   calls `aws ec2 describe-instances` filtering for
   `tag:pcs-cluster-id=<my cluster>` + `tag:directory-role=server` +
   `instance-state-name=running`, and reads the matching instance's
   `PrivateIpAddress`. The `pcs-cluster-id` filter scopes the lookup to **this
   cluster only**, so multiple PCS clusters can share one VPC without their
   compute nodes finding the wrong cluster's LDAP server. (`CLUSTER_ID` is
   passed from `${ClusterId}` in UserData; the script aborts client setup if it
   is empty, rather than risk matching another cluster's server.)
3. That IP becomes the SSSD `ldap_uri` (`ldap://<login-ip>`). SSSD on the
   compute node then resolves users from the login node's slapd.

Implications to be aware of:

- **Compute nodes need `ec2:DescribeInstances`** in their instance role (the
  cluster IAM role already grants it). Without it, discovery fails and the node
  boots without LDAP (check `/var/log/directory-setup.log` for
  `could not discover directory server IP`).
- **The login (server) node must be running before a compute node boots** for
  discovery to succeed. In normal deploy order the login node group comes up
  first; a compute node that scales up later simply queries the
  already-running server.
- **If the login node is replaced**, its new instance re-tags itself
  `directory-role=server` and re-attaches to the same `/home/ldap-db`, so newly
  booting compute nodes discover the new IP automatically. Already-running
  compute nodes keep their cached `ldap_uri`; they pick up the new IP on their
  next boot (or after an SSSD reconfigure).
- **An explicit override exists**: set `LDAP_SERVER_URI` (or `DIRECTORY_DNS_IPS`,
  for the future managed-directory path) in the client's environment to skip the
  tag lookup entirely — used by the SimpleAD/ManagedAD extension and handy for
  debugging.

---

## Enabling multi-user

### Option 1: deploy-all (recommended)

```bash
aws cloudformation create-stack \
  --stack-name my-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-2b \
    ParameterKey=DirectoryService,ParameterValue=OpenLDAP-LoginNode \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

That's it. The login node will have slapd running and compute nodes will be
configured as LDAP clients automatically at first boot.

### Option 2: modular deployment

Pass these to your add-cng.yaml stacks:
- Login CNG: `DirectoryService=OpenLDAP-LoginNode`, `DirectoryRole=server`
- Compute CNG: `DirectoryService=OpenLDAP-LoginNode`, `DirectoryRole=client`

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `DirectoryService` | `none` | Set to `OpenLDAP-LoginNode` to enable multi-user |
| `DirectoryRole` | `none` | Auto-set by deploy-all: `server` for login, `client` for compute |
| `DirectoryDomainSuffix` | `dc=cluster,dc=internal` | LDAP base DN (change only if you need a different domain) |

---

## Day-to-day operations

All commands below run on the **login node** as root (`sudo`).

### Getting the admin password

The LDAP admin password is auto-generated at cluster creation and stored in
AWS Systems Manager Parameter Store:

```bash
CLUSTER_ID=<from stack output, e.g. pcs_abc123>

aws ssm get-parameter \
  --name "/pcs/${CLUSTER_ID}/ldap/admin-password" \
  --with-decryption \
  --query 'Parameter.Value' \
  --output text
```

> **If SSM is empty** (instance role lacked the permission at first boot):
> ```bash
> sudo cat /home/ldap-db/.admin-password
> ```

Store this password somewhere safe — you'll need it for all user management
operations.

---

### Adding a user

**Using the helper script** (recommended):

```bash
# Usage: ldap-add-user <username> <uid> <gid> [ssh-public-key]
sudo LDAP_ADMIN_PASSWORD="<password>" ldap-add-user alice 10001 3000
```

This creates the user with:
- Username: `alice`
- UID: `10001` (pick a unique number in range 10001–59999)
- GID: `3000` (= `clusterusers` group, the default)
- Home directory: `/home/alice` (auto-created on first login)
- Shell: `/bin/bash`
- A random initial password (printed to stdout)

**With an SSH key** (user can log in immediately):

```bash
sudo LDAP_ADMIN_PASSWORD="<password>" ldap-add-user alice 10001 3000 "ssh-rsa AAAA... alice@laptop"
```

**Verifying the user was created:**

```bash
# On login node
getent passwd alice
# Expected: alice:*:10001:3000:alice:/home/alice:/bin/bash

id alice
# Expected: uid=10001(alice) gid=3000(clusterusers) groups=3000(clusterusers)
```

---

### Adding multiple users (batch)

Create a file `users.txt`:
```
alice 10001 3000 ssh-rsa AAAA...
bob   10002 3000 ssh-rsa BBBB...
carol 10003 3000
```

Then:
```bash
while read name uid gid key; do
  sudo LDAP_ADMIN_PASSWORD="<password>" ldap-add-user "$name" "$uid" "$gid" "$key"
done < users.txt
```

---

### Listing all users

```bash
# Simple list
ldapsearch -x -H ldap://localhost -b "ou=People,dc=cluster,dc=internal" \
  "(objectClass=posixAccount)" uid uidNumber | grep -E "^uid:|^uidNumber:"

# Or just use getent (shows all LDAP users + system users)
getent passwd | awk -F: '$3 >= 10000 {print $1, $3, $6}'
```

---

### Deleting a user

```bash
ADMIN_PW="<password>"
ldapdelete -x -H ldap://localhost \
  -D "cn=admin,dc=cluster,dc=internal" \
  -w "$ADMIN_PW" \
  "uid=alice,ou=People,dc=cluster,dc=internal"
```

Also remove from Slurm accounting:
```bash
sacctmgr -i remove user alice
```

The user's home directory (`/home/alice`) is NOT deleted automatically.
Remove it manually if needed:
```bash
sudo rm -rf /home/alice
```

---

### Resetting a user's password

```bash
ADMIN_PW="<password>"
NEW_PW="temporary-password-123"

ldappasswd -x -H ldap://localhost \
  -D "cn=admin,dc=cluster,dc=internal" \
  -w "$ADMIN_PW" \
  -s "$NEW_PW" \
  "uid=alice,ou=People,dc=cluster,dc=internal"

echo "New password for alice: $NEW_PW"
```

Tell the user to change it after login:
```bash
# User runs this after logging in
ldappasswd -x -H ldap://localhost \
  -D "uid=alice,ou=People,dc=cluster,dc=internal" \
  -W -s "my-new-password" \
  "uid=alice,ou=People,dc=cluster,dc=internal"
```

---

### Creating groups

```bash
ADMIN_PW="<password>"

ldapadd -x -H ldap://localhost \
  -D "cn=admin,dc=cluster,dc=internal" \
  -w "$ADMIN_PW" << EOF
dn: cn=ml-team,ou=Groups,dc=cluster,dc=internal
objectClass: posixGroup
cn: ml-team
gidNumber: 3001
memberUid: alice
memberUid: bob
EOF
```

### Adding a user to a group

```bash
ldapmodify -x -H ldap://localhost \
  -D "cn=admin,dc=cluster,dc=internal" \
  -w "$ADMIN_PW" << EOF
dn: cn=ml-team,ou=Groups,dc=cluster,dc=internal
changetype: modify
add: memberUid
memberUid: carol
EOF
```

---

## Slurm accounting

PCS manages the Slurm accounting database internally. You just need to register
users and accounts:

```bash
export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH

# Create a Slurm account (typically one per team or project)
sacctmgr -i add account default Description="Default account"

# Add users to the account
sacctmgr -i add user alice Account=default
sacctmgr -i add user bob Account=default

# Verify
sacctmgr show user
```

> **Note:** if `AccountingPolicyEnforcement=none` (the default), users can
> submit jobs even without being registered in `sacctmgr`. Registration is
> needed for fairshare/priority and for `sacct` history to show the user name.

---

## Verifying users on compute nodes

After adding a user, verify they're visible on compute nodes:

```bash
export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH

# Single node
srun -N 1 -n 1 -p cpu1 bash -c 'getent passwd alice; id alice'

# All nodes
srun -N 4 -n 4 -p cpu1 bash -c 'echo "$(hostname): $(id alice)"'
```

If a user isn't visible yet (SSSD cache delay, typically <5 sec):
```bash
srun -N 1 -n 1 -p cpu1 bash -c 'sudo sss_cache -E; sleep 2; getent passwd alice'
```

---

## Running jobs as a specific user

Users log in to the login node and submit jobs normally:

```bash
# User 'alice' logs in via SSH and runs:
srun -p cpu1 -N 1 -n 1 bash -c 'whoami; hostname'
sbatch --partition=cpu1 my-training.sbatch
```

The job runs as `alice` (uid=10001) on the compute node. The user's home
directory `/home/alice` is visible on the compute node (shared OpenZFS).

---

## Troubleshooting

### "User not found" on compute node

```bash
# Check SSSD is running on compute
srun -N 1 -n 1 bash -c 'systemctl status sssd | head -3'

# Check LDAP connectivity from compute
srun -N 1 -n 1 bash -c 'ldapsearch -x -H ldap://<login-ip> -b dc=cluster,dc=internal uid=alice'

# Force cache refresh
srun -N 1 -n 1 bash -c 'sudo sss_cache -E; sudo systemctl restart sssd'
```

### slapd not running on login node

```bash
sudo systemctl status slapd
sudo journalctl -u slapd -n 20
# Check install log
cat /var/log/directory-setup.log
```

### "Invalid credentials" when running ldap commands

You're using the wrong admin password. Retrieve it from SSM or the fallback
file (see [Getting the admin password](#getting-the-admin-password)).

### Home directory not created

```bash
# Check pam_mkhomedir is configured
grep pam_mkhomedir /etc/pam.d/common-session
# Expected: session optional pam_mkhomedir.so skel=/etc/skel umask=0022

# Manually create (should auto-create on next login)
sudo mkdir -p /home/alice
sudo chown alice:clusterusers /home/alice
```

### New compute node doesn't resolve users

New compute nodes boot with the latest LaunchTemplate version, which includes
SSSD client setup. If a node was launched before `DirectoryService` was enabled
(e.g. during a stack update), it won't have SSSD. Terminate the node and let
PCS replace it with a new one.

---

## UID/GID conventions

| Range | Purpose |
|---|---|
| 0–999 | System users (do not use) |
| 1000 | `ubuntu` (DLAMI default user) |
| 3000 | `clusterusers` group (default GID for new users) |
| 3001+ | Additional groups (create as needed) |
| 10001–59999 | LDAP user UIDs |

**Always specify UIDs explicitly** when creating users. This ensures
consistency across all nodes and NFS mounts. Do not rely on auto-increment.

---

## Data persistence and backup

| Data | Location | Survives node replacement? | Survives stack delete? |
|---|---|---|---|
| LDAP database | `/home/ldap-db/` (shared OpenZFS) | ✅ | ❌ (FSx deleted) |
| User home directories | `/home/<user>/` (shared OpenZFS) | ✅ | ❌ (FSx deleted) |
| Admin password | SSM Parameter Store | ✅ | ✅ |

### Backup

```bash
# Export LDAP database to a file (run periodically via cron)
sudo slapcat -l /home/ldap-backup-$(date +%Y%m%d).ldif
```

### Restore (on a fresh login node)

```bash
sudo systemctl stop slapd
sudo slapadd -l /home/ldap-backup-YYYYMMDD.ldif
sudo chown -R openldap:openldap /home/ldap-db
sudo systemctl start slapd
```

---

## Access methods

| Method | Best for | Setup required |
|---|---|---|
| **Direct SSH** (port 22) | Multi-user teams, VS Code/JupyterLab | SG rule opening port 22 to a CIDR |
| **SSH over SSM** | Security-sensitive environments | IAM credentials + SSM plugin per user |
| **SSM Session Manager** | Admin-only access | IAM credentials only |

For multi-user clusters, **Direct SSH** is recommended. Users connect with
their SSH key that was added during user creation:

```bash
ssh alice@<login-node-public-ip>
```

---

## Template structure

```
deploy-all.yaml
├─► cluster.yaml         (IAM role with ssm:PutParameter for /pcs/<id>/ldap/*)
├─► add-cng.yaml (login) → DirectoryRole=server → setup-directory.sh server
│                           (installs slapd + configures SSSD locally)
└─► add-cng.yaml (compute) → DirectoryRole=client → setup-directory.sh client
                              (installs SSSD, discovers login node IP)
```

---

## Upgrading to AWS Simple AD (future)

If you outgrow OpenLDAP (need HA, >50 users, Kerberos), the
`DirectoryService` parameter is designed for extension:

```yaml
DirectoryService: SimpleAD   # future AllowedValue
```

Migration path:
1. Export users: `slapcat > users.ldif`
2. Deploy Simple AD (separate stack, requires 2 AZs)
3. Import users
4. Redeploy cluster with `DirectoryService=SimpleAD`
5. Decommission slapd

See [docs/ROADMAP.md](./ROADMAP.md) for tracking.
