# Add Users to SageMaker HyperPod Slurm Clusters

Standalone extension script that adds users to SageMaker HyperPod Slurm clusters.
Designed for use with the `OnInitComplete` lifecycle configuration (AMI-based clusters).

## What It Does

1. **Creates POSIX users** on all nodes with consistent UIDs
2. **Sets up home directories** on the shared filesystem (FSx for Lustre or OpenZFS)
3. **Generates SSH keypairs** on the shared filesystem for passwordless inter-node SSH
4. **Registers users in Slurm accounting** so they can submit jobs (controller only)

## Prerequisites

- A SageMaker HyperPod Slurm cluster with AMI-based configuration
- The cluster execution role must have S3 read access to the scripts bucket
- A shared filesystem (FSx for Lustre or OpenZFS) is recommended but not required.
  Without one, home directories and SSH keys are local to each node, and
  passwordless cross-node SSH will not work.

## Configuration

The script supports two input formats. Copy the appropriate sample file,
rename it, and edit it with your users:

| Input File | Sample File | Format | Features |
|------------|-------------|--------|----------|
| `shared_users.txt` | `shared_users_sample.txt` | CSV | Legacy format, backward compatible with base LCS |
| `shared_users.yaml` | `shared_users_sample.yaml` | YAML | Groups, per-group Slurm accounts, filesystem mounts |

If both `shared_users.txt` and `shared_users.yaml` are present, the text file
takes precedence (even if empty) for backward compatibility.

### Legacy Format (shared_users.txt)

Plain CSV |-- one user per line: `username,uid,home_directory`

```
alice,2001,/fsx/alice
bob,2002,/fsx/bob
carol,2003,/fsx/carol
```

This is the same format used by the base lifecycle scripts. All users are added
to the `root` Slurm account. The home directory path (third field) is accepted
for compatibility but the script auto-detects the shared filesystem mount.

### YAML Simple Format (shared_users.yaml)

A flat list of users |-- all get global defaults:

```yaml
users:
  - username: alice
    uid: 2001
  - username: bob
    uid: 2002

# Optional global defaults:
# shared_filesystem_mount: /fsx    # Default: auto-detect
# slurm_account: root              # Default: root
```

### YAML Groups Format (shared_users.yaml)

Users organized into groups with per-group Slurm accounts and filesystem mounts.
Groups are purely organizational |-- they do not create Linux groups. Per-group
settings override the global defaults.

```yaml
# Optional global defaults (used when a group doesn't specify its own):
# shared_filesystem_mount: /fsx
# slurm_account: root

groups:
  - name: research
    slurm_account: research
    users:
      - username: alice
        uid: 2001
      - username: bob
        uid: 2002

  - name: platform
    slurm_account: platform
    users:
      - username: carol
        uid: 3001
      - username: dave
        uid: 3002
```

### Configuration Reference (YAML format)

| Field | Required | Scope | Default | Description |
|-------|----------|-------|---------|-------------|
| `users` | Yes* | Global |  | Simple format: flat list of users. |
| `groups` | Yes* | Global |  | Groups format: list of user groups. |
| `groups[].name` | Yes | Group |  | Group name (organizational only, no Linux group created). |
| `groups[].users` | Yes | Group |  | Users in this group. |
| `groups[].slurm_account` | No | Group | `root` | Slurm account for this group. Overrides global default. |
| `groups[].shared_filesystem_mount` | No | Group | auto-detect | Home directory path for this group. Overrides global default. |
| `shared_filesystem_mount` | No | Global | auto-detect | Default home directory path for all users/groups. |
| `slurm_account` | No | Global | `root` | Default Slurm account for all users/groups. |

\* Use either `users` or `groups`, not both.

### Filesystem Auto-Detection

The script automatically detects which shared filesystem to use for home directories,
matching the base lifecycle scripts' behavior:

1. **OpenZFS at `/home`** (NFS mount) |-- preferred when present
2. **FSx for Lustre at `/fsx`** |-- fallback

When both OpenZFS and Lustre are mounted, home directories go on OpenZFS (`/home/<user>`)
and a data directory is also created on Lustre (`/fsx/<user>`) for training data and
checkpoints.

Per-group `shared_filesystem_mount` overrides the auto-detection for that group.

### Choosing UIDs

- Use UIDs >= 2000 to avoid conflicts with system users
- UIDs must be unique across the cluster and across all groups
- UIDs must be consistent across all nodes (this script handles that)

## Deployment

1. Copy the appropriate sample file and edit it:
   ```bash
   # For YAML format:
   cp shared_users_sample.yaml shared_users.yaml
   # Or for legacy format:
   cp shared_users_sample.txt shared_users.txt
   ```
2. Upload to S3:
   ```bash
   aws s3 cp add-users/ s3://<your-bucket>/add-users/ --recursive
   ```
3. Create or update your cluster with `OnInitComplete`:
   ```json
   {
       "LifeCycleConfig": {
           "OnInitComplete": "add_users.sh",
           "SourceS3Uri": "s3://<your-bucket>/add-users/"
       }
   }
   ```

## Adding Users After Cluster Creation

`OnInitComplete` only runs during node provisioning. To add users to existing nodes:

1. Update your user file with the new users (keep existing users in the file)
2. Upload to S3:
   ```bash
   aws s3 cp add-users/shared_users.yaml s3://<your-bucket>/add-users/shared_users.yaml
   ```
3. SSM into the controller and pull scripts to the shared filesystem:
   ```bash
   sudo mkdir -p /fsx/cluster-scripts/add-users
   sudo aws s3 cp s3://<your-bucket>/add-users/ /fsx/cluster-scripts/add-users/ --recursive
   sudo chmod +x /fsx/cluster-scripts/add-users/*.sh
   ```
4. Run on the controller:
   ```bash
   sudo bash /fsx/cluster-scripts/add-users/add_users.sh
   ```
5. Run on all compute nodes via srun:
   ```bash
   sudo srun --partition=<partition-name> bash /fsx/cluster-scripts/add-users/add_users.sh
   ```

Existing users are skipped |-- only new users are created. The scripts are fully idempotent.

## Execution Order

The scripts run in dependency order on every node, once per group:

```
add_users.sh (entrypoint)
  |-- For each group:
      |-- create_posix_users.sh   All nodes: create POSIX users
      |-- setup_home_dirs.sh      All nodes: home dirs on shared FS
      |-- setup_ssh_keys.sh       All nodes: SSH keys on shared FS
      |-- setup_slurm_accounts.sh Controller only: Slurm accounting
```

## How It Works

### Node Type Detection

The script auto-detects whether it's running on a controller, compute, or login
node by checking Slurm daemons:
- `slurmctld` running |-- controller
- `slurmd` running + hostname in `sinfo` |-- compute
- `slurmd` running + hostname not in `sinfo` |-- login

### SSH Key Sharing

SSH keypairs are generated on the shared filesystem (e.g., `/fsx/<username>/.ssh/`).
The first node to run creates the keypair; subsequent nodes find it already there.
This ensures all nodes share the same keys for passwordless SSH.

### Idempotency

All scripts are idempotent |-- safe to re-run without errors or duplicates:
- Users that already exist are skipped
- Home directories that already exist are left alone
- SSH keys that already exist are not regenerated
- Slurm accounting associations that already exist are not duplicated

## Files

| File | Description |
|------|-------------|
| `add_users.sh` | Entrypoint script (OnInitComplete target) |
| `create_posix_users.sh` | Creates POSIX users |
| `setup_home_dirs.sh` | Sets up home directories on shared filesystem |
| `setup_ssh_keys.sh` | Generates SSH keypairs on shared filesystem |
| `setup_slurm_accounts.sh` | Registers users in Slurm accounting |
| `shared_users_sample.yaml` | Sample YAML config (copy to `shared_users.yaml`) |
| `shared_users_sample.txt` | Sample legacy CSV config (copy to `shared_users.txt`) |
| `.gitattributes` | Enforces LF line endings |

## Troubleshooting

### Logs

All output is logged to `/var/log/provision/add_users.log` on each node.

### Common Issues

**"No shared filesystem detected"**: This is a warning, not an error. The script
will still create users with local `/home` directories. However, passwordless
cross-node SSH will not work. For shared home directories and SSH keys, configure
FSx for Lustre or OpenZFS in your cluster's `InstanceStorageConfigs`.

**"Neither slurmctld nor slurmd is running"**: The script must run after
Slurm starts. With `OnInitComplete`, this is guaranteed. If running manually,
ensure Slurm is started first.

**Users can't submit jobs**: Check that Slurm accounting is set up by running
`sacctmgr show user` on the controller. If the user is missing, re-run the
script on the controller.

**SSH between nodes fails**: Verify keys are on the shared filesystem:
`ls -la /fsx/<username>/.ssh/`. If keys are on local disk instead, the home
directory setup may not have completed before SSH key generation.
