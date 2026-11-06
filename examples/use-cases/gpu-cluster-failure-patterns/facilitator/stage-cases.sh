#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Stage the diagnosis case bundle read-only on the coordinator.
#
#   sudo bash stage-cases.sh <source-bundle-dir> [dest]
#
# Default dest is /opt/aim344/cases. The bundle is world-readable and
# root-owned: participants read it, nobody edits it in place, and answers go to
# each participant's own directory instead.
#
# Ordering, which is the point of this script rather than an implementation
# detail: a candidate is verified against its build manifest while it is still
# private, and only a candidate that passes is exposed at the participant path.
# A candidate that fails verification is never mounted, so "refused" means the
# material was withheld and not that a nonzero status was returned after
# participants could already read it. If the exposed path cannot be verified by
# reading back through it, the mount is withdrawn and the path is left
# unavailable rather than serving unverified files.
#
# "Private" is a permission on the candidate, not only the absence of a mount.
# Review finding R10: an earlier version created the master root and every
# candidate directory 0755, so a participant could read the unverified tree at
# its own path under the master root while $dest still served the last good
# tree, and an interrupted copy left that tree readable indefinitely. The master
# root is therefore 0700 and each candidate directory is created 0700; a
# candidate becomes group/other-traversable only after it has been verified,
# and even then it is reachable only through the bind mount, because path
# resolution to the mount point does not traverse the master root. An
# interrupted or killed run leaves an unreadable directory behind, marked
# incomplete so the next run removes it.
#
# This deliberately does NOT alias or wrap any diagnostic command. A participant
# who runs nvidia-smi or sinfo during the case work gets the real tool talking
# about this real cluster.
set -euo pipefail

[[ $(id -u) == 0 ]] || { echo 'Run this with sudo on the coordinator node.' >&2; exit 1; }

source_dir=${1:?source bundle directory}
dest=${2:-/opt/aim344/cases}

[[ -d $source_dir ]] || { echo "Not a directory: $source_dir" >&2; exit 2; }
manifest=$source_dir/MANIFEST.json
[[ -s $manifest ]] || { echo "Missing manifest: $manifest" >&2; exit 2; }

# The staged tree must outlive both a coordinator reboot and a stop/start.
# Measured on the coordinator: /opt/aim344 is a bind mount from
# /dev/mapper/vg.01-lv_ephemeral, the instance-store LVM, so anything written
# straight to $dest is lost when the instance stops. /opt itself, and /var, are
# on the root EBS volume (/dev/nvme0n1p1). So the durable master lives on the
# root volume and $dest is a read-only bind mount of one of its versioned trees.
master_root=${AIM344_CASES_MASTER:-/var/lib/aim344/cases}
# Overridable only so the negative controls can exercise the real code path
# without editing the host's fstab. Production runs leave it at the default.
fstab=${AIM344_FSTAB:-/etc/fstab}

case $dest in
    /opt/aim344/*) ;;
    *) echo "Refusing an unexpected destination: $dest" >&2; exit 2 ;;
esac

# Create the master root before asking which filesystem it is on: findmnt -T on
# a path that does not exist yet returns nothing, and under `set -e` that aborts
# the script with no message at all.
#
# 0700, not 0755: this directory holds candidate trees that have not been
# verified yet. With it traversable, /var/lib/aim344/cases/tree-* was a second
# pathname to the same bytes the bind mount deliberately withholds. Applied with
# install -d on every run, so a host staged by an earlier version is tightened
# the next time this script runs.
install -d -o root -g root -m 0700 "$master_root"
master_device=$(findmnt -T "$master_root" -n -o SOURCE | sed 's/\[.*//')
[[ -n $master_device ]] || { echo "Cannot determine the filesystem for $master_root" >&2; exit 2; }
if [[ $master_device == *ephemeral* || $master_device == *dlami* ]]; then
    echo "Refusing: the durable master would sit on instance storage ($master_device)." >&2
    echo 'Set AIM344_CASES_MASTER to a path on the root volume.' >&2
    exit 2
fi
echo "== durable master root $master_root on $master_device"

# What is exposed right now, so a refusal can say what was preserved.
active_link=$master_root/current
active_before=''
if [[ -L $active_link ]]; then active_before=$(readlink -f "$active_link" || true); fi
mounted_before=no
if findmnt -T "$dest" -n -o TARGET 2>/dev/null | grep -qx "$dest"; then mounted_before=yes; fi
printf '   currently exposed: mounted=%s tree=%s\n' "$mounted_before" "${active_before:-none}"

# Remove any candidate an earlier run left behind unverified. The marker file
# sits beside the tree rather than inside it, because verify() treats any file
# the manifest does not describe as a finding and would fail on its own marker.
#
# Measured, not assumed: a SIGKILL to this script does not kill the `install`
# children that xargs has already started, so an abandoned tree can still be
# growing when the next run tries to purge it. A single `rm -rf` then fails with
# "Directory not empty" and, under `set -e`, took the whole next staging run down
# with it — the facilitator could not restage at all after one killed run
# (rework6-d3-repro.sh, 4 of 12 attempts). So the purge retries while a writer
# drains, and a purge that still cannot finish is a warning rather than a fatal
# error: the tree is 0700 and marked incomplete, so it is unreachable by a
# participant either way, and refusing to stage the accepted bundle over it would
# be the worse failure.
purge_tree() {
    local tree=$1 marker=$2
    local attempt
    for attempt in $(seq 1 60); do
        rm -rf "$tree" 2>/dev/null
        [[ -e $tree ]] || { rm -f "$marker"; return 0; }
        sleep 0.5
    done
    chmod -R 0700 "$tree" 2>/dev/null || true
    echo "   WARNING: could not remove $tree (a writer from the interrupted run may still hold it)." >&2
    echo "   It stays mode 0700 and marked incomplete, so no participant can read it." >&2
    return 1
}
shopt -s nullglob
for stale in "$master_root"/tree-*.incomplete; do
    stale_tree=${stale%.incomplete}
    echo "== removing an unverified tree left by an interrupted run: $stale_tree"
    purge_tree "$stale_tree" "$stale" || true
done
shopt -u nullglob

# The candidate is built under a fresh versioned directory. It is private until
# it has been verified: mode 0700 under a 0700 master root, so no participant can
# reach it by any pathname, and nothing mounts it. $dest keeps serving whatever
# it was serving before.
candidate=$master_root/tree-$(date -u +%Y%m%dT%H%M%SZ)-$$
if [[ -e $candidate ]]; then echo "Refusing: $candidate already exists" >&2; exit 2; fi
install -d -o root -g root -m 0700 "$candidate"
: > "$candidate.incomplete"
exposed=no
cleanup_unexposed() {
    # A kill during the copy or the verification must not leave a readable tree
    # behind. Until $exposed is yes the candidate is 0700 and unreachable anyway;
    # this removes it rather than relying on the next run's purge.
    if [[ $exposed == no && -d $candidate ]]; then
        rm -rf "$candidate" "$candidate.incomplete"
        echo "Interrupted before exposure; removed the unverified candidate $candidate" >&2
    fi
}
trap cleanup_unexposed INT TERM HUP
echo "== building candidate $candidate (private, mode 0700, not exposed yet)"

# Copy content only; never preserve source ownership, which may be a build user.
# The build manifest is a facilitator artefact used to verify staging; it is not
# part of what a participant reads, so it is deliberately not copied in.
# Directories are 0700 for the same reason the candidate root is: an unverified
# tree must not be readable at its own path. They are opened up after
# verification, immediately before the mount.
(cd "$source_dir" && find . -type d -print0 | xargs -0 -I{} install -d -o root -g root -m 0700 "$candidate/{}")
(cd "$source_dir" && find . -type f ! -name MANIFEST.json -print0 \
    | xargs -0 -I{} install -o root -g root -m 0444 "{}" "$candidate/{}")

verify() {
    # $1 manifest, $2 tree root, $3 label. Reads the tree it is given, so the
    # same check runs against the private candidate and again through the
    # exposed mount.
    python3 - "$1" "$2" "$3" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path, root, label = sys.argv[1:4]
manifest = json.loads(pathlib.Path(manifest_path).read_text())
root = pathlib.Path(root)

problems = []
checked = 0
for entry in manifest['files']:
    # Paths in the manifest are relative to the bundle root, so the same
    # manifest verifies the tree on any host. An absolute build-host path here
    # would make this check fail everywhere except the build machine, which
    # would look like a refusal without testing anything.
    relative = pathlib.PurePosixPath(entry['path'])
    if relative.is_absolute():
        problems.append(f'manifest records an absolute path: {relative}')
        continue
    staged = root / relative
    if not staged.is_file():
        problems.append(f'missing: {relative}')
        continue
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    if digest != entry['sha256']:
        problems.append(f'hash differs: {relative}\n  manifest {entry["sha256"]}\n  tree     {digest}')
        continue
    mode = staged.stat().st_mode & 0o777
    if mode != 0o444:
        problems.append(f'not read-only: {relative} mode {mode:o}')
        continue
    checked += 1

# Anything present that the manifest does not describe is also a problem: it
# would mean an unreviewed file reached the participants.
described = {str(pathlib.PurePosixPath(entry['path'])) for entry in manifest['files']}
for path in root.rglob('*'):
    if path.is_file():
        relative = str(path.relative_to(root))
        if relative not in described:
            problems.append(f'present but not in the manifest: {relative}')

if problems:
    print(f'VERIFICATION FAILED ({label})')
    for problem in problems:
        print(f'  {problem}')
    raise SystemExit(1)
print(f'verified {checked} files against the manifest in {label}, all read-only')
PY
}

echo "== verifying the candidate against the build manifest, BEFORE exposing it"
if ! verify "$manifest" "$candidate" 'private candidate'; then
    echo 'Refusing to expose an unverified bundle.' >&2
    rm -rf "$candidate" "$candidate.incomplete"
    if [[ $mounted_before == yes ]]; then
        echo "   $dest still serves the previously verified tree: ${active_before:-unknown}" >&2
        findmnt -n -o SOURCE,TARGET,OPTIONS "$dest" | sed 's/^/   /' >&2
    else
        echo "   $dest was not exposed before this run and is still not exposed." >&2
    fi
    exit 4
fi

# Verified, so it may now be traversed. Directory modes are opened up only here,
# after the content check passed: before this line the tree had the bytes but not
# the permissions, which is what "private until verified" has to mean. The master
# root stays 0700, so the only path a participant can use is the bind mount.
find "$candidate" -type d -exec chmod 0755 {} +
rm -f "$candidate.incomplete"

echo "== exposing the verified candidate at $dest"
install -d -o root -g root -m 0755 "$dest"
if [[ $mounted_before == yes ]]; then
    umount "$dest"
fi
mount --bind "$candidate" "$dest"
mount -o remount,bind,ro "$dest"
findmnt -n -o SOURCE,TARGET,OPTIONS "$dest" | sed 's/^/   /'

withdraw() {
    echo "Withdrawing $dest rather than serving unverified material." >&2
    umount "$dest" 2>/dev/null || true
    # The candidate failed a check after it was mounted, so it is not material
    # anyone should be able to reach. Tighten it before removing it, so a failure
    # of rm still leaves it unreadable rather than a readable rejected tree.
    if [[ -d $candidate ]]; then
        chmod -R 0700 "$candidate" 2>/dev/null || true
        rm -rf "$candidate" "$candidate.incomplete"
        echo "   removed the rejected candidate $candidate" >&2
    fi
    if [[ -n $active_before && -d $active_before ]]; then
        if mount --bind "$active_before" "$dest" && mount -o remount,bind,ro "$dest"; then
            echo "   restored the last verified tree: $active_before" >&2
            exit 5
        fi
        echo "   could not restore $active_before; $dest left unavailable" >&2
    else
        echo "   no previously verified tree to restore; $dest left unavailable" >&2
    fi
    exit 5
}

# Read the exposed path back. The mount options string is not the property that
# matters; what participants get when they read through $dest is.
echo "== verifying what is readable through $dest"
verify "$manifest" "$dest" "exposed path $dest" || withdraw
exposed=yes

# Prove the read-only property rather than trusting the mount options.
if : > "$dest/.write-probe" 2>/dev/null; then
    rm -f "$dest/.write-probe"
    echo "Refusing: root could write into the read-only bind mount." >&2
    withdraw
fi
echo '   confirmed: even root cannot write through the bind mount'

echo "== confirming the bundle is not writable, as an unprivileged account"
# Use an existing unprivileged account if one is present; this is an assertion
# about the staged permissions, so it must actually attempt the write.
probe_user=$(getent passwd | awk -F: '$3 >= 1000 && $3 < 65534 {print $1}' | head -1)
if [[ -n $probe_user ]]; then
    if runuser -u "$probe_user" -- test -r "$dest/README.md"; then
        echo "   $probe_user can read the bundle"
    else
        echo "   Refusing: $probe_user cannot read the bundle" >&2
        withdraw
    fi
    if runuser -u "$probe_user" -- bash -c ": > '$dest/README.md'" 2>/dev/null; then
        echo "   Refusing: $probe_user was able to write into the bundle" >&2
        withdraw
    fi
    echo "   $probe_user cannot write into the bundle"
    # R10: the mount is one pathname to these bytes; the master root is another.
    # Exposing the accepted tree must not expose the directory that also holds
    # candidates. Both halves are asserted, because a check that only tests the
    # reachable path would pass on the version this replaces.
    if runuser -u "$probe_user" -- test -x "$master_root" 2>/dev/null; then
        echo "   Refusing: $probe_user can traverse the candidate master root $master_root" >&2
        withdraw
    fi
    if runuser -u "$probe_user" -- test -r "$candidate/README.md" 2>/dev/null; then
        echo "   Refusing: $probe_user can read the accepted tree at its own path $candidate" >&2
        withdraw
    fi
    echo "   $probe_user cannot reach $master_root, so candidates have no second pathname"
else
    echo '   no unprivileged account present to test with; permissions set but untested'
fi

# Record which tree is live, and persist the bind mount so a coordinator reboot
# does not silently leave the participants with an empty directory. The fstab
# line names the versioned tree, so a reboot restores exactly the tree that was
# verified rather than whatever a later run left in a shared directory.
ln -sfn "$candidate" "$active_link"
tmp_fstab=$(mktemp)
grep -v " $dest " "$fstab" > "$tmp_fstab" || true
printf '%s %s none bind,ro 0 0\n' "$candidate" "$dest" >> "$tmp_fstab"
install -o root -g root -m 0644 "$tmp_fstab" "$fstab"
rm -f "$tmp_fstab"
echo "   $fstab entry now names $candidate"

# Keep the previously verified tree so a later failed candidate has something to
# fall back to, and prune anything older than that pair. Trees still marked
# incomplete are left to the purge at the top of the next run: they may still have
# a writer attached, and a failed rm here would abort the script after the mount is
# already live.
mapfile -t trees < <(find "$master_root" -mindepth 1 -maxdepth 1 -type d -name 'tree-*' | sort)
if (( ${#trees[@]} > 2 )); then
    for old in "${trees[@]:0:${#trees[@]}-2}"; do
        [[ $old == "$candidate" || $old == "$active_before" ]] && continue
        [[ -e $old.incomplete ]] && continue
        purge_tree "$old" "$old.incomplete" && echo "   pruned superseded tree $old"
    done
fi

echo "== staged inventory"
find "$dest" -type f | sort | sed 's/^/   /'
printf 'Case bundle staged read-only at %s from %s\n' "$dest" "$candidate"
