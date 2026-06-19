# User Management Guide

This guide is written for **cluster administrators who may not be familiar with
LDAP**. It covers the day-to-day operations of managing users on a PCS
reference architecture cluster with `DirectoryService=OpenLDAP-LoginNode`.

By default, the cluster runs as a single `ubuntu` user. When multi-user is
enabled, an OpenLDAP directory runs on the login node and provides centralized
POSIX user accounts visible on all nodes via SSSD.

---

## Quick reference (common tasks)

Run on the login node. `$ADMIN_PW` is the LDAP admin password from SSM (see
[Getting the admin password](#getting-the-admin-password)); the `LDAP_*` env
vars must be passed **inline to `sudo`** (`sudo VAR=... cmd`) — `sudo -E` alone
drops them under the default `env_reset`/`secure_path`.

> **Slurm path matches `SlurmVersion`.** The `sacctmgr` / `sbatch` paths below use
> `slurm-25.11` (the default). If you deployed with `SlurmVersion=25.05`, replace
> `slurm-25.11` with `slurm-25.05` in every path (or just run `export
> PATH=/opt/aws/pcs/scheduler/slurm-$(ls /opt/aws/pcs/scheduler | sed 's/slurm-//')/bin:$PATH`
> once and drop the absolute path).

| Task | Command (run on the login node) |
|---|---|
| Add a user | `sudo LDAP_ADMIN_PASSWORD="$ADMIN_PW" ldap-add-user.sh alice 10001 3000` |
| List all users | `ldapsearch -x -H ldap://localhost -b ou=People,dc=cluster,dc=internal uid` |
| Delete a user | `ldapdelete -x -H ldap://localhost -D cn=admin,dc=cluster,dc=internal -w "$ADMIN_PW" uid=alice,ou=People,dc=cluster,dc=internal` then `sudo sss_cache -E` |
| Reset a user's password | `ldappasswd -x -H ldap://localhost -D cn=admin,dc=cluster,dc=internal -w "$ADMIN_PW" -s NEWPASS uid=alice,ou=People,dc=cluster,dc=internal` |
| Add user to Slurm accounting | `sudo /opt/aws/pcs/scheduler/slurm-25.11/bin/sacctmgr -i add user alice Account=ml-team` (root = accounting admin) |
| Verify user on compute node | `srun -N1 -n1 -p cpu1 id alice` |

`ldap-add-user.sh` is a helper script installed on the login node at
`/usr/local/bin/` (it wraps `ldapadd`+`ldappasswd` so you don't need the LDAP
syntax). List/delete/reset use the raw `ldap*` tools directly — the full
commands are documented in each section below.

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
  booting compute nodes discover the new IP automatically. The admin password is
  **preserved** across replacement (the new login node reuses the existing
  `/pcs/<id>/ldap/admin-password` from SSM rather than regenerating it), so cached
  admin passwords keep working and the DB stays accessible.
  **Already-running compute nodes, however, keep their cached `ldap_uri`** (the
  old login node's private IP, which the replacement no longer has). Cached users
  still resolve via SSSD, but **uncached/new users and group expansion fail** on
  those nodes until you refresh them. After a login-node replacement, run on every
  running compute node:
  ```bash
  srun -N <nodes> -n <nodes> bash -c 'sudo sed -i "s#ldap_uri = .*#ldap_uri = ldap://<new-login-ip>#" /etc/sssd/sssd.conf && sudo sss_cache -E && sudo systemctl restart sssd'
  ```
  or simply let PCS replace the compute nodes (terminate them; the replacements
  discover the new IP on boot). A stable endpoint that removes this manual step is
  tracked in [ROADMAP.md](./ROADMAP.md) ("Stable LDAP endpoint across login-node
  replacement").
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

Pass these to your `add-cng.yaml` stacks:
- Login CNG: `DirectoryRole=server`, `DirectoryDomainSuffix=dc=cluster,dc=internal`
- Compute CNG: `DirectoryRole=client`, `DirectoryDomainSuffix=dc=cluster,dc=internal`

> **IAM profile — give the login CNG the login profile.** `cluster.yaml` outputs two
> instance profiles: `InstanceProfileArn` (compute) and `LoginInstanceProfileArn` (login,
> which additionally grants read+decrypt of the OpenLDAP admin secret). When deploying
> modularly with multi-user enabled, pass `IamProfileArn=<LoginInstanceProfileArn>` to the
> **login** CNG and `IamProfileArn=<InstanceProfileArn>` to the **compute** CNGs. Giving
> the login node the plain compute profile makes its OpenLDAP setup fail (it can't write
> the admin password to SSM). `deploy-all` wires this automatically.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `DirectoryService` | `none` | (deploy-all only) Set to `OpenLDAP-LoginNode` to enable multi-user; deploy-all derives each CNG's `DirectoryRole` from it |
| `DirectoryRole` | `none` | (add-cng) `server` for the login CNG, `client` for compute CNGs |
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
# Usage: ldap-add-user.sh <username> <uid> <gid> [ssh-public-key]
sudo LDAP_ADMIN_PASSWORD="<password>" ldap-add-user.sh alice 10001 3000
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
sudo LDAP_ADMIN_PASSWORD="<password>" ldap-add-user.sh alice 10001 3000 "ssh-rsa AAAA... alice@laptop"
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
  sudo LDAP_ADMIN_PASSWORD="<password>" ldap-add-user.sh "$name" "$uid" "$gid" "$key"
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

**Invalidate the SSSD cache** so the deletion takes effect immediately instead
of lingering until the cache entry's TTL expires. SSSD caches lookups (so users
still resolve during a brief LDAP outage), which means a freshly-deleted user
keeps resolving via `getent` until the cache is cleared. Run on the login node
**and** any running compute node:
```bash
sudo sss_cache -E      # invalidate all cached entries (needs the sssd-tools package, pre-installed)
# across compute nodes:
srun -N <nodes> -n <nodes> bash -c 'sudo sss_cache -E'
```

Also remove from Slurm accounting:
```bash
sudo /opt/aws/pcs/scheduler/slurm-25.11/bin/sacctmgr -i remove user alice
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

PCS manages the Slurm accounting database internally (enable it with
`ManagedAccounting=enabled` at deploy time). You just need to register users and
accounts.

> **Run `sacctmgr` add/modify/remove as `root`.** In PCS managed accounting the
> Administrator is `root` — the default `ubuntu` user is not an accounting admin,
> so `sacctmgr -i add ...` as `ubuntu` fails with *"Only
> admins/operators/coordinators can add accounts"*. Use `sudo` with the full
> path (the Slurm bin dir isn't on root's `PATH` by default):

```bash
SACCTMGR=/opt/aws/pcs/scheduler/slurm-25.11/bin/sacctmgr

# Create a Slurm account (typically one per team or project)
sudo $SACCTMGR -i add account ml-team Description="ML Team"

# Add LDAP users to the account
sudo $SACCTMGR -i add user alice Account=ml-team
sudo $SACCTMGR -i add user bob Account=ml-team

# Verify (read-only — works as any user with the Slurm bin on PATH)
sudo $SACCTMGR show user alice bob format=User,Account,DefaultAccount
```

Read-only `sacct` / `sreport` / `sacctmgr show ...` work as `ubuntu`; only the
mutating `sacctmgr` verbs need `root`.

> **Two reporting gotchas (verified in testing — they are tool behaviour, not bugs):**
>
> 1. **`sacct --state=...` needs an explicit end time.** A state filter without
>    `-E now` (or `--endtime`) silently returns **nothing** — e.g.
>    `sacct -X --state=FAILED -S 2026-01-01` shows no rows even when failed jobs
>    exist. Always pair it: `sacct -X -a --state=FAILED -S <start> -E now`.
>    (`sacct -j <id>` always shows the true state, useful to cross-check.)
> 2. **`sreport` usage lags behind `sacct` by up to an hour.** `sreport`
>    (`AccountUtilizationByUser`, `topusage`, cluster `Utilization`) reads
>    slurmdbd's **periodic usage rollup**, which runs hourly — so right after jobs
>    finish, `sreport ... Used` reads **zero** (even at `-t Seconds`) until the next
>    rollup boundary. `sacct` reads the live job table and is immediate. For
>    up-to-the-second per-job/-user accounting use `sacct`; use `sreport` for
>    settled historical utilization.

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

> **After a login-node replacement, the public IP and SSH host key change.**
> The login node has no Elastic IP, so a replacement (or stop/start) gives it a
> **new public IP** — re-fetch it from the EC2 console or
> `aws ec2 describe-instances`. A *replacement* (not a stop/start) is a brand-new
> instance, so its **SSH host key also changes**: users will see
> `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!` and must clear the old key
> (`ssh-keygen -R <old-ip-or-hostname>`) before reconnecting. Tell users to expect
> this whenever the login node is replaced. (SSH-over-SSM avoids the IP problem
> since it targets the instance ID, not an IP.)

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
