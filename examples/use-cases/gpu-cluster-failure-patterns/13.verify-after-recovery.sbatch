#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Participant verification after a device round: take a fresh allocation, run the
# named suite's Check 5 once from the coordinator, and complete the storage
# workload against the recreated private fixture.
#
#   sbatch --wait /opt/aim344/device-recovery/13.verify-after-recovery.sbatch
#
# Runtime ready and workload verified are separate results. This script produces
# the second one; it does not decide device health, which the suite owns.
#SBATCH --job-name=aim344-verify
#SBATCH --partition=gpu-g7
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:25:00
# The batch host may be either assigned node, and the staged tree is root-owned
# there, so an output file under it cannot be created and the job fails with
# ExitCode 0:53 before the first line runs. Write to a participant-writable
# directory that exists on both nodes.
#SBATCH --chdir=/var/tmp
#SBATCH --output=/var/tmp/aim344-verify-%j.log
set -uo pipefail
base=${AIM344_BASE:-/opt/aim344/device-recovery}
export PATH=/opt/amazon/efa/bin:/opt/aws/pcs/scheduler/slurm-25.05/bin:$PATH
run=${AIM344_VERIFY_NAME:-verify-$SLURM_JOB_ID}
out=${AIM344_RESULTS_DIR:-/var/tmp/aim344-results}/$run
mkdir -p "$out"
status=0
# How many EFA devices each allocated node must have carried traffic on. This is
# a declared expectation rather than a count taken from the node at judging time:
# reading it from the node would make a node that came back with one device look
# complete, which is the failure this whole exercise is about. The suite's own
# check 2 uses the same expectation of 2 for this pair.
expected_efa=${AIM344_EXPECTED_EFA_DEVICES:-2}
printf '== output log: /var/tmp/aim344-verify-%s.log on %s\n' "$SLURM_JOB_ID" "$(hostname)"
printf '   this log lives on the batch host only. The retrieval step at the end of\n'
printf '   this script copies every result, including this log, to /var/tmp on\n'
printf '   each allocated node so you can read it from your own terminal.\n'

printf '== allocation: %s nodes on %s\n' "${SLURM_JOB_NUM_NODES:-?}" "${SLURM_JOB_NODELIST:-?}"
# The node set this run must account for. Judging per device is only meaningful
# against the list of nodes that were actually allocated: a verdict computed from
# whichever files happen to be visible cannot notice a node that produced none.
expected_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
expected_node_count=$(printf '%s\n' "$expected_nodes" | grep -c .)
printf '   expected nodes (%s): %s\n' "$expected_node_count" "$(printf '%s ' $expected_nodes)"
printf '   expected EFA devices carrying traffic per node: %s\n' "$expected_efa"
# Each counter record identifies itself by hostname, because that is what the
# producer knows (efa-counters.py prints `host`), while the allocation is expressed
# in Slurm node names. Resolve the mapping here, from the scheduler, so the
# judgement can tie a record to an allocated node without depending on the order
# in which concurrent ranks' output arrives.
#
# That dependency was a real defect, found by running this: srun multiplexes the
# ranks' stdout, so on job 104 BOTH `AIM344_DELTA_NODE=` markers arrived before
# BOTH records. Attributing each record to the marker above it therefore assigned
# both to one node and reported the other as having produced nothing, failing a run
# whose two nodes had in fact each moved ~11 GB on each of two devices. A marker
# line is not a reliable envelope for a separate line from the same rank.
node_map=''
for node in $expected_nodes; do
  host=$(scontrol show node "$node" | tr ' ' '\n' | sed -n 's/^NodeHostName=//p' | head -1)
  [ -n "$host" ] || host=$node
  node_map="$node_map$node=$host
"
done
printf '   node name to hostname: %s\n' "$(printf '%s' "$node_map" | tr '\n' ' ')"
srun -N2 -n2 --ntasks-per-node=1 -c192 --mpi=none --cpu-bind=none \
  bash -c 'printf "  %s (%s): GPUs=%s EFA=%s\n" "$(hostname)" "${SLURMD_NODENAME:-?}" \
    "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" \
    "$(ls /sys/class/infiniband | wc -l)"'

printf '== EFA counters before the collective\n'
export RUN_NAME=$run
# No pipe here on purpose. `producer | tee file` reports tee's status, so a rank
# whose snapshot failed looked like a success; the file is written directly and
# the producer's own status is what this step returns.
srun -N2 -n2 --ntasks-per-node=1 --mpi=none --cpu-bind=none \
  bash -c 'd='"$out"'/$(hostname); mkdir -p "$d";
    python3 '"$base"'/companion/efa-counters.py snapshot "$d/before"' || status=1

printf '== Check 5: multi-node NCCL collective, invoked once from the coordinator\n'
# Call the suite's own entry point rather than the qualification wrapper, which
# also syncs to an evidence bucket the participant has no reason to write.
suite=${AIM344_SUITE_ENTRY:-$base/candidate-v2/validation/gpu-cluster-healthcheck/gpu-healthcheck.sh}
printf '   suite_entry=%s\n' "$suite"
sha256sum "$suite" | sed 's/^/   suite_entry_sha256=/'
[ -f "$base/candidate-v2/REVISION" ] && \
  sed 's/^/   suite_revision=/' "$base/candidate-v2/REVISION"
# No sudo. The participant is not granted any sudo rule, so a sudo here made this
# entry point unusable as authorized: the one scoped rule belongs to the control
# account and permits only the session helper. Check 5 does not need root either,
# which is established by execution rather than assumed: the same suite entry ran
# to [PASS] as unprivileged aim344-t1 inside the participant's own allocation
# (participant-revision/runs/verify-participant-check5d/output.log:27).
printf '   invoked_as=%s (uid %s), sudo is not used here\n' "$(id -un)" "$(id -u)"
export NCCL_CONTAINER=/opt/aim344/nccl-baseline.sqsh NCCL_TIMEOUT=600
bash "$suite" --check 5 --verbose --timeout 660 --results-dir "$out/check5" \
  2>&1 | tee "$out/check5.log"
check5=${PIPESTATUS[0]}
printf '   check5_exit_code=%s (dimensionless)\n' "$check5"
[ "$check5" -eq 0 ] || status=1

printf '== Check 5 raw correctness rows, from the suite own raw capture\n'
# A verdict line alone does not show the measurement. Print the size/correctness
# columns the suite recorded, so a green summary cannot stand in for the data.
raw=$out/check5/nccl-allreduce-raw.txt
if [ -f "$raw" ]; then
  printf '   raw_capture=%s\n' "$raw"
  grep -E '^[[:space:]]*[0-9]+[[:space:]]' "$raw" | sed 's/^/   /'
  grep -E 'Avg bus bandwidth|Out of bounds values' "$raw" | sed 's/^/   /'
else
  printf '   raw_capture=ABSENT at %s; correctness rows are UNVALIDATED\n' "$raw"
  status=1
fi

printf '== EFA counter deltas per device, collected from every allocated node\n'
# Each rank writes its own record to its own node-local results directory AND
# prints it here, prefixed with the Slurm node name it came from. The printed copy
# is what the judgement below reads, because the judgement runs on the batch host
# and that host's filesystem holds only its own record: globbing there judged one
# node and silently ignored the other (measured on job 89, where both hosts'
# records were printed but only the target's two devices were judged,
# participant-revision/runs/rework2-read-job89/output.log:55-59).
#
# Again no pipe: the producer's status is preserved, and `srun` returns non-zero
# if any task failed, so a rank that produced nothing fails this step instead of
# disappearing from a glob.
deltas=$out/efa-deltas-all-nodes.txt
srun -N2 -n2 --ntasks-per-node=1 --mpi=none --cpu-bind=none \
  bash -c 'd='"$out"'/$(hostname);
    printf "AIM344_DELTA_NODE=%s host=%s\n" "${SLURMD_NODENAME:-unknown}" "$(hostname)";
    python3 '"$base"'/companion/efa-counters.py delta "$d/before" > "$d/efa-deltas.json";
    rc=$?;
    cat "$d/efa-deltas.json";
    exit $rc' 2>&1 | tee "$deltas"
delta_producer=${PIPESTATUS[0]}
printf '   delta_producer_exit_code=%s (dimensionless)\n' "$delta_producer"
[ "$delta_producer" -eq 0 ] || status=1

printf '== per-device EFA byte deltas, judged per node and per device\n'
AIM344_EXPECTED_NODES="$expected_nodes" AIM344_EXPECTED_EFA="$expected_efa" \
AIM344_NODE_MAP="$node_map" \
python3 - "$deltas" <<'PYDELTA' || status=1
"""Judge the collected records against the node set that was allocated.

Three failures this replaces, all of which passed before:
  a node that produced no record at all (the batch host's glob never saw it),
  a record whose deltas_bytes mapping was empty (a file existing was taken as
  evidence that a device moved bytes), and
  a node reporting fewer devices than this round expects.

Records are keyed by the `host` field the producer itself wrote, mapped back to
Slurm node names through the scheduler's own NodeHostName. The earlier version
attributed each record to the AIM344_DELTA_NODE marker printed above it, which
looked equivalent and is not: srun multiplexes concurrent ranks, so on job 104
both markers arrived before both records and one node was reported as silent while
it had in fact moved ~11 GB on each of its two devices.
"""
import json
import os
import re
import sys

source = open(sys.argv[1], encoding='utf-8', errors='replace').read()
expected_nodes = [n for n in os.environ['AIM344_EXPECTED_NODES'].split() if n]
expected_efa = int(os.environ['AIM344_EXPECTED_EFA'])
host_of = {}
for entry in os.environ.get('AIM344_NODE_MAP', '').split():
    if '=' in entry:
        node, host = entry.split('=', 1)
        host_of[node] = host
print(f'   expected_nodes={",".join(expected_nodes)} '
      f'expected_efa_devices_per_node={expected_efa}')
print(f'   host mapping={host_of}')

# Every JSON record in the stream, keyed by the host that produced it. The node
# markers are still printed for a human reader; they are deliberately not what the
# attribution depends on.
by_host = {}
for line in source.split('\n'):
    stripped = line.strip()
    if not stripped.startswith('{'):
        continue
    try:
        record = json.loads(stripped)
    except ValueError:
        continue
    if not isinstance(record.get('deltas_bytes'), dict):
        continue
    host = record.get('host')
    if not host:
        continue
    by_host.setdefault(host, []).append(record)

failures = []
claimed = set()
for node in expected_nodes:
    host = host_of.get(node, node)
    found = by_host.get(host, [])
    if not found:
        failures.append(f'{node} ({host}): produced NO counter record; its '
                        f'traffic is UNVALIDATED')
        continue
    claimed.add(host)
    record = found[-1]
    per_device = {}
    for counter, value in record['deltas_bytes'].items():
        match = re.search(r'/sys/class/infiniband/([^/]+)/', counter)
        device = match.group(1) if match else counter
        per_device.setdefault(device, {})[counter.rsplit('/', 1)[-1]] = value
    if not per_device:
        failures.append(f'{node}: its record contains no device counters at all')
        continue
    for device in sorted(per_device):
        counters = per_device[device]
        moved = counters.get('tx_bytes', 0) + counters.get('rx_bytes', 0)
        verdict = 'moved bytes' if moved > 0 else 'ZERO BYTES'
        print(f"   {node} ({host}) {device}: "
              f"tx_bytes={counters.get('tx_bytes')} "
              f"rx_bytes={counters.get('rx_bytes')} "
              f"rdma_write_bytes={counters.get('rdma_write_bytes')} -> {verdict}")
        if moved <= 0:
            failures.append(f'{node} {device}: carried no traffic')
    if len(per_device) != expected_efa:
        failures.append(f'{node}: {len(per_device)} device(s) reported traffic, '
                        f'expected {expected_efa}')

unexpected = sorted(set(by_host) - claimed)
if unexpected:
    failures.append('records arrived from hosts outside this allocation: '
                    + ','.join(unexpected))
if failures:
    print('   per-device traffic FAILED:')
    for failure in failures:
        print(f'     {failure}')
    raise SystemExit(1)
print(f'   every expected node reported {expected_efa} device(s) with traffic')
PYDELTA

printf '== storage workload against the recreated private fixture\n'
# Confirm the fixture is usable by THIS account on every allocated node before
# running the workload. The fixture was configured for `ubuntu` while the
# participant ran as aim344-t1, so /run/aim344-checkpoints was mode 0700 owned by
# a different user and the container's /checkpoints was unreadable. That is a
# real failure of the exercise, not a detail, and it must be visible here rather
# than surfacing as an opaque workload error.
printf '   fixture ownership on each allocated node\n'
srun -N2 -n2 --ntasks-per-node=1 -c192 --mpi=none --cpu-bind=none \
  bash -c 'printf "   %s: %s readable=%s writable=%s\n" "$(hostname)" \
    "$(stat -c "owner=%U mode=%a" /run/aim344-checkpoints 2>&1)" \
    "$( [ -r /run/aim344-checkpoints ] && echo yes || echo NO )" \
    "$( [ -w /run/aim344-checkpoints ] && echo yes || echo NO )"' || status=1
srun -N2 -n2 --ntasks-per-node=1 -c192 --mpi=none --cpu-bind=none \
  bash -c 'test -r /run/aim344-checkpoints && test -w /run/aim344-checkpoints'
fixture=$?
printf '   fixture_accessible_on_all_nodes_exit_code=%s (dimensionless)\n' "$fixture"

# One function, used by every exit path below, so a run that stops early still
# leaves the participant able to read what it did produce. An early `exit` that
# skipped retrieval would reproduce R7 for exactly the failing rounds whose
# evidence matters most.
retrieve_results() {
  local overall=$1 storage_state=$2
  printf '== verification summary\n'
  printf '   results=%s\n' "$out"
  printf '   check5=%s storage=%s overall=%s\n' "$check5" "$storage_state" "$overall"
  {
    printf 'job=%s\n' "$SLURM_JOB_ID"
    printf 'batch_host=%s\n' "$(hostname)"
    printf 'nodelist=%s\n' "$SLURM_JOB_NODELIST"
    printf 'expected_nodes=%s\n' "$(printf '%s,' $expected_nodes)"
    printf 'expected_efa_devices_per_node=%s\n' "$expected_efa"
    printf 'check5_exit_code=%s\n' "$check5"
    printf 'storage=%s\n' "$storage_state"
    printf 'overall=%s\n' "$overall"
    printf 'completed_at=%s\n' "$(date -u +%FT%TZ)"
  } > "$out/summary.txt"
  # The job's own log is written by slurmd on the batch host as this script runs,
  # so the copy taken here necessarily stops a few lines short of the end. It is
  # included anyway because it holds the Check 5 output, and the authoritative
  # verdict travels separately in summary.txt.
  cp -f "/var/tmp/aim344-verify-$SLURM_JOB_ID.log" \
        "$out/job-log-as-of-retrieval.txt" 2>/dev/null
  local bundle=/var/tmp/aim344-verify-$SLURM_JOB_ID-results.tar
  tar -cf "$bundle" -C "$(dirname "$out")" "$(basename "$out")" || {
    printf '   RESULT RETRIEVAL FAILED: could not package %s\n' "$out"
    return 1
  }
  local want
  want=$(sha256sum "$bundle" | cut -d' ' -f1)
  printf '   result_bundle=%s sha256=%s %s B\n' \
    "$bundle" "$want" "$(stat -c %s "$bundle")"
  # Copy it to every allocated node, so the participant reads it from their own
  # terminal whichever node Slurm chose as the batch host. This uses plain srun
  # and the job the participant already holds: no sudo, no ssh, no facilitator.
  # sbcast would be the obvious tool and does not work here -- measured as
  # aim344-t1 inside their own job: "sbcast: error: Slurm job 98 lookup error:
  # Unspecified error", rc=255
  # (participant-revision/runs/rework5-r7-read-probe-log-target/output.log).
  # A plain stdin redirect through srun does work, and lands byte-identical
  # participant-owned copies on both nodes
  # (participant-revision/runs/rework5-r7-read-transfer-log-target/output.log).
  srun -N"$expected_node_count" -n"$expected_node_count" --ntasks-per-node=1 \
    --mpi=none --cpu-bind=none \
    bash -c 'cat > '"$bundle"'.incoming && mv '"$bundle"'.incoming '"$bundle" \
    < "$bundle"
  local copied=$?
  printf '   result_distribution_exit_code=%s (dimensionless)\n' "$copied"
  # Verify the bytes on every node rather than trusting the copy: a stale or
  # truncated file at the destination is worse than none, because it reads as a
  # result. This prints one line per node and fails the retrieval if any digest
  # differs.
  local check
  check=$(srun -N"$expected_node_count" -n"$expected_node_count" \
    --ntasks-per-node=1 --mpi=none --cpu-bind=none \
    bash -c 'printf "   retrieved on %s (%s): %s\n" "$(hostname)" \
      "${SLURMD_NODENAME:-?}" "$(sha256sum '"$bundle"' 2>&1 | cut -d" " -f1)"' 2>&1)
  printf '%s\n' "$check"
  local matched
  matched=$(printf '%s\n' "$check" | grep -c "$want")
  printf '   nodes_with_matching_bundle=%s of %s\n' "$matched" "$expected_node_count"
  if [ "$copied" -ne 0 ] || [ "$matched" -ne "$expected_node_count" ]; then
    printf '   RESULT RETRIEVAL FAILED: the bundle is not readable and identical\n'
    printf '   on every allocated node, so this run cannot be inspected from the\n'
    printf '   coordinator. Treat the run as unverified.\n'
    return 1
  fi
  printf '   To read this result from your own terminal, on the coordinator:\n'
  # The independent login coordinator is not an allocated GPU node. Retrieve
  # through a fresh participant allocation after this job ends, not root SSH.
  local retrieval_node
  retrieval_node=$(printf '%s\n' "$expected_nodes" | head -1)
  printf '     srun --partition=%s --nodelist=%s --nodes=1 --ntasks=1 --cpus-per-task=1 --mpi=none cat %s > ~/aim344-verify-%s-results.tar\n' \
    "${SLURM_JOB_PARTITION:-gpu-g7}" "$retrieval_node" "$bundle" "$SLURM_JOB_ID"
  printf '     printf "%%s  %%s\\n" %s ~/aim344-verify-%s-results.tar | sha256sum -c -\n' \
    "$want" "$SLURM_JOB_ID"
  printf '     mkdir -p ~/aim344-verify-%s && tar -xf %s -C ~/aim344-verify-%s\n' \
    "$SLURM_JOB_ID" "~/aim344-verify-$SLURM_JOB_ID-results.tar" "$SLURM_JOB_ID"
  printf '     cat ~/aim344-verify-%s/%s/summary.txt\n' "$SLURM_JOB_ID" "$run"
  return 0
}

if [ "$fixture" -ne 0 ]; then
  printf '   the private checkpoint fixture is not usable by %s on every node;\n' "$(id -un)"
  printf '   the distributed storage workload cannot be attributed to a recovered node\n'
  printf '   Stopping before the workload rather than running it. A rank that cannot\n'
  printf '   read /checkpoints does not fail fast: it blocks in rendezvous until the\n'
  printf '   job hits its time limit, which spends the whole allocation and buries\n'
  printf '   the permission cause under a TCPStore timeout. Measured: job 86 ran\n'
  printf '   ~10 more minutes and was CANCELLED DUE TO TIME LIMIT, and the summary\n'
  printf '   below never printed.\n'
  retrieve_results 1 not-attempted
  exit 1
fi
export GPUS_PER_NODE=8 MASTER_PORT=$((29600 + SLURM_JOB_ID % 100)) FI_EFA_USE_HUGE_PAGE=0
# An IP address, not the Slurm node name. `scontrol show hostnames` returns the
# scheduler's names (gpu-g7-1), and those do not resolve inside the container:
# measured on job 87, where every rank logged "The IPv6 network addresses of
# (gpu-g7-1, 29687) cannot be retrieved (gai error: -3)" and then sat in
# rendezvous until the job hit its 25-minute time limit. The known-good runs on
# this cluster pass a literal address for exactly this reason. NodeAddr is what
# Slurm itself uses to reach the node, so read it rather than assuming the name
# resolves.
first_node=$(printf '%s\n' "$expected_nodes" | head -1)
MASTER_ADDR=$(scontrol show node "$first_node" \
  | tr ' ' '\n' | sed -n 's/^NodeAddr=//p' | head -1)
if [ -z "${MASTER_ADDR:-}" ]; then
  printf '   could not resolve a rendezvous address for %s; not starting the\n' "$first_node"
  printf '   workload, because a rank that cannot rendezvous blocks until the job\n'
  printf '   time limit rather than failing\n'
  retrieve_results 1 not-attempted
  exit 1
fi
export MASTER_ADDR
printf '   rendezvous: first_node=%s MASTER_ADDR=%s MASTER_PORT=%s\n' \
  "$first_node" "$MASTER_ADDR" "$MASTER_PORT"
# Confirm the address actually answers from every allocated node before spending
# the allocation on a workload that would otherwise hang.
srun -N2 -n2 --ntasks-per-node=1 -c192 --mpi=none --cpu-bind=none \
  bash -c 'getent hosts '"$MASTER_ADDR"' >/dev/null || \
    printf "   %s: cannot resolve %s\n" "$(hostname)" '"$MASTER_ADDR"'
    timeout 5 bash -c "</dev/tcp/'"$MASTER_ADDR"'/22" 2>/dev/null \
      && printf "   %s: reaches %s\n" "$(hostname)" '"$MASTER_ADDR"' \
      || printf "   %s: cannot reach %s on a known port\n" "$(hostname)" '"$MASTER_ADDR"''
srun -N2 -n2 --ntasks-per-node=1 -c192 --mpi=none --cpu-bind=none \
  --container-image=/opt/aim344/aim344.sqsh \
  --container-name="aim344_verify_$SLURM_JOB_ID" \
  --container-writable --no-container-mount-home \
  --container-env=MASTER_ADDR,MASTER_PORT,GPUS_PER_NODE,FI_EFA_USE_HUGE_PAGE \
  --container-mounts="$base/companion:/opt/aim344:ro,/run/aim344-checkpoints:/checkpoints" \
  bash /opt/aim344/torch-node.sh storage 2>&1 | tee "$out/storage.log"
storage=${PIPESTATUS[0]}
printf '   storage_exit_code=%s (dimensionless)\n' "$storage"
[ "$storage" -eq 0 ] || status=1

retrieve_results "$status" "$storage" || status=1
exit "$status"
