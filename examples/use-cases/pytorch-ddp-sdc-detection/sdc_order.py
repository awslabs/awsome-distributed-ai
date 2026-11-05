"""SDC checks for PyTorch DistributedDataParallel (DDP) training, in the recommended order.

Per step: (1) forward and backward, with a comm hook that records each rank's local gradient;
(2) non-finite gradient check; (3) ring replay audit on audit steps, or when check 1 fires;
(4) gate: on a compute-path alarm the optimizer step isn't applied; the reduced gradient is also
checked for NaN or Inf before the update; (5) weight fingerprint comparison after the step, every
N steps and on every checkpoint step; (6) checkpoint only if no check fired.

Fingerprints are two wrapping int64 sums per tensor: one of the raw bit patterns, and one of the
bit patterns weighted by position. A change to one or two elements of a tensor always changes
them (for tensors under 2**32 elements with elements of up to 4 bytes). Larger changes can, in
principle, cancel out, so equal fingerprints are strong evidence, not proof, of equal tensors.
"""
import os
import socket
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.attention import SDPBackend, sdpa_kernel

_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


@torch.no_grad()
def fingerprints(tensors, chunk=1 << 24):
    """Two int64 per tensor: wrapping sums of the raw bit patterns, unweighted and weighted by
    (1-based) position. A None entry counts as all zeros. Returns a [len(tensors), 2] tensor."""
    dev = next((t.device for t in tensors if t is not None), None)
    if dev is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    out = torch.zeros(len(tensors), 2, dtype=torch.int64, device=dev)
    for k, t in enumerate(tensors):
        if t is None:
            continue
        bits = t.detach().contiguous().view(-1).view(_INT_VIEW[t.element_size()])
        for i in range(0, bits.numel(), chunk):  # chunked to bound temporary memory
            b = bits[i:i + chunk].to(torch.int64)
            pos = torch.arange(i + 1, i + 1 + b.numel(), dtype=torch.int64, device=dev)
            out[k, 0] += b.sum()
            out[k, 1] += (b * pos).sum()
    return out


def find_divergent_ranks(params, dp_group):
    """Call on every DP rank at the same step, after optimizer.step(). A rank is a suspect when
    its copy of a tensor differs from a strict majority (more than half) of the copies. With no
    strict majority for some tensor, "suspects" is None: a disagreement, but no attribution."""
    local = fingerprints(params).flatten()
    n = dist.get_world_size(dp_group)
    out = torch.empty(n * local.numel(), dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(out, local, group=dp_group)
    table = out.view(n, -1, 2).cpu()  # [dp_size, num_tensors, 2]
    differs = (table != table[0]).any(dim=2).any(dim=0)
    if not differs.any():
        return {"diverged": False, "suspects": []}
    suspects = set()
    for k in differs.nonzero().flatten().tolist():
        copies = [tuple(row) for row in table[:, k].tolist()]
        value, count = Counter(copies).most_common(1)[0]
        if 2 * count <= n:
            return {"diverged": True, "suspects": None}  # no strict majority for tensor k
        suspects.update(r for r, c in enumerate(copies) if c != value)
    return {"diverged": True,
            "suspects": sorted(dist.get_global_rank(dp_group, r) for r in suspects)}


def check_topology(dp_group, offset, node=None):
    """Call on every DP rank. Checks that every ring replayer (rank r), its target (r + offset),
    and the confirming GPU (r + 2 * offset) sit on three different nodes, and raises ValueError
    if not. `node` defaults to $NODE_NAME (set it from spec.nodeName), then the hostname.
    Returns every DP rank's node name."""
    node = node or os.environ.get("NODE_NAME") or socket.gethostname()
    n = dist.get_world_size(dp_group)
    nodes = [None] * n
    dist.all_gather_object(nodes, node, group=dp_group)
    bad = [r for r in range(n)
           if len({nodes[r], nodes[(r + offset) % n], nodes[(r + 2 * offset) % n]}) < 3]
    if bad:
        raise ValueError(f"peer_offset={offset}: the replayer, target, and confirmer share a node "
                         f"for DP ranks {bad[:8]}; every ring pair needs three different nodes")
    return nodes


class LocalStats:
    """Comm-hook state: this rank's local (pre-all-reduce) gradient statistics for one step."""

    def __init__(self, dp_group, params):
        self.dp_group = dp_group
        self.record_fingerprints = False
        self.row = {p.data_ptr(): i for i, p in enumerate(params)}  # parameter -> row of fp
        self.sq = []
        self.fp = torch.zeros(len(params), 2, dtype=torch.int64, device=params[0].device)

    def clear(self):
        self.sq.clear()
        self.fp.zero_()


def local_stats_hook(stats, bucket):
    stats.sq.append(bucket.buffer().float().pow(2).sum())
    if stats.record_fingerprints:  # per parameter, before allreduce_hook divides in place
        rows = [stats.row[p.data_ptr()] for p in bucket.parameters()]
        stats.fp[rows] = fingerprints(bucket.gradients())
    return allreduce_hook(stats.dp_group, bucket)


def attach(ddp_model, dp_group):
    """Register the hook once per DDP model; returns the state that train() reads."""
    stats = LocalStats(dp_group, list(ddp_model.module.parameters()))
    ddp_model.register_comm_hook(state=stats, hook=local_stats_hook)
    return stats


def nonfinite_ranks(stats):
    """After backward: global ranks whose local gradient norm is NaN or Inf."""
    local = torch.stack(stats.sq).sum().sqrt().reshape(1)
    n = dist.get_world_size(stats.dp_group)
    gathered = [torch.empty_like(local) for _ in range(n)]
    dist.all_gather(gathered, local, group=stats.dp_group)
    bad = ~torch.isfinite(torch.cat(gathered).cpu())
    return [dist.get_global_rank(stats.dp_group, r) for r in bad.nonzero().flatten().tolist()]


@contextmanager
def deterministic():
    """Deterministic kernels for one step, restoring the caller's settings afterwards. An op with
    no deterministic implementation raises instead of warning, so an audit never compares
    non-deterministic results. Needs CUBLAS_WORKSPACE_CONFIG=:4096:8 at startup."""
    prev = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        with sdpa_kernel(SDPBackend.MATH):  # flash, memory-efficient, cuDNN attention aren't
            yield
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])


def _bits(x):
    """A scalar's raw bit pattern, in its own dtype, as int64."""
    return x.detach().reshape(1).view(_INT_VIEW[x.element_size()]).to(torch.int64)


def _replay(module, compute_loss, batch, with_grads):
    """Recompute one micro-batch with the unwrapped module. Afterwards, restores the averaged
    gradients, the module's buffers, and the RNG state, so the replay leaves no trace."""
    params = list(module.parameters())
    saved = [p.grad for p in params]
    buffers = [b.detach().clone() for b in module.buffers()]
    rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    for p in params:
        p.grad = None
    loss = compute_loss(module, batch)
    loss.backward()
    fp = fingerprints([p.grad for p in params]) if with_grads else \
        torch.zeros(len(params), 2, dtype=torch.int64, device=loss.device)
    for p, g in zip(params, saved):
        p.grad = g
    with torch.no_grad():
        for b, s in zip(module.buffers(), buffers):
            b.copy_(s)
    torch.set_rng_state(rng[0])
    torch.cuda.set_rng_state(rng[1])
    return torch.cat([_bits(loss), fp.flatten()])  # [loss bits, per-parameter fingerprints]


def ring_audit(module, compute_loss, rebuild_batch, step, stats, local_loss, offset, with_grads):
    """Every DP rank replays the micro-batch of the rank `offset` positions ahead (on another
    node when offset = GPUs per node). Call after backward, before optimizer.step(). Compares the
    loss bits and, if with_grads, every parameter's local-gradient fingerprints. Returns the
    mismatched (target, replayer) pairs as global ranks, and the gathered table."""
    g = stats.dp_group
    n = dist.get_world_size(g)
    me = dist.get_group_rank(g, dist.get_rank())
    target = dist.get_global_rank(g, (me + offset) % n)
    replay = _replay(module, compute_loss, rebuild_batch(step, target), with_grads)
    orig_fp = stats.fp if with_grads else torch.zeros_like(stats.fp)
    v = torch.cat([_bits(local_loss), orig_fp.flatten(), replay])
    out = torch.empty(n * v.numel(), dtype=torch.int64, device=v.device)
    dist.all_gather_into_tensor(out, v, group=g)
    table = out.view(n, 2, -1).cpu()  # per DP rank: [own result, its replay of its target]
    pairs = [(dist.get_global_rank(g, (r + offset) % n), dist.get_global_rank(g, r))
             for r in range(n) if not torch.equal(table[(r + offset) % n, 0], table[r, 1])]
    return pairs, table


def localize(pairs):
    """A GPU in two mismatched pairs (its own micro-batch, and the one it replayed) is the culprit.
    Pairs that share no culprit stay ambiguous: either side may have computed wrong."""
    hits = Counter(r for p in pairs for r in p)
    culprits = sorted(r for r, c in hits.items() if c >= 2)
    ambiguous = [p for p in pairs if not set(p) & set(culprits)]
    return culprits, ambiguous


def confirm(module, compute_loss, rebuild_batch, step, pair, table, stats, offset, with_grads):
    """Settle one ambiguous (target, replayer) pair in place: the GPU `offset` positions after the
    target (on a third node, which check_topology verifies) and the target itself replay the
    target's micro-batch again. Call on every DP rank."""
    g = stats.dp_group
    n = dist.get_world_size(g)
    target, replayer = pair
    ti, ri = dist.get_group_rank(g, target), dist.get_group_rank(g, replayer)
    third = dist.get_global_rank(g, (ti + offset) % n)
    me = dist.get_rank()
    v = torch.zeros(2, table.shape[2], dtype=torch.int64,
                    device=torch.device("cuda", torch.cuda.current_device()))
    for slot, who in ((0, third), (1, target)):
        if me == who:
            v[slot] = _replay(module, compute_loss, rebuild_batch(step, target), with_grads)
    dist.all_reduce(v, group=g)  # each slot is written by exactly one rank
    third_r, self_r = v.cpu()
    orig, peer = table[ti, 0], table[ri, 1]
    if torch.equal(third_r, orig):
        return {"pair": pair, "third": third, "suspect": replayer, "kind": "replayer computed wrong"}
    if torch.equal(third_r, peer):
        kind = ("persistent" if torch.equal(self_r, orig) else
                "transient" if torch.equal(self_r, third_r) else "unclear")
        return {"pair": pair, "third": third, "suspect": target, "kind": kind}
    return {"pair": pair, "third": third, "suspect": None, "kind": "unattributable"}


@dataclass
class Schedule:
    audit_every: int = 10            # every rank's micro-batch is replayed on these steps
    deterministic_audits: bool = True  # also compare gradients (catches backward-pass faults)
    check_every: int = 10            # weight fingerprint comparison
    ckpt_every: int = 100
    check_on_checkpoint: bool = True
    peer_offset: int = 8             # GPUs per node: the replayer sits on another node
    max_grad_norm: float = 1.0


def _window(last_pass, step, start_step):
    return [start_step if last_pass is None else last_pass + 1, step]


def train(ddp_model, stats, optimizer, batches, compute_loss, rebuild_batch, save_checkpoint,
          sched, start_step=0):
    """Runs until `batches` is exhausted (returns None) or until the first alarm (returns it).
    On a compute-path or reduced-gradient alarm, the step that raised it isn't applied; earlier
    steps may have been.

    Checkpoints are tagged with the last step whose audit passed and the last step whose weight
    comparison passed. An alarm carries both, plus suspect_window: the steps since the last pass
    of the check that fired, in which a fault that was active on every step could have started."""
    module = ddp_model.module
    params = list(ddp_model.parameters())
    dp_group = stats.dp_group
    check_topology(dp_group, sched.peer_offset)
    last_audit = last_weights = None
    for step, batch in enumerate(batches, start_step):
        audit = step % sched.audit_every == 0                               # 0. decide first
        det = audit and sched.deterministic_audits
        stats.clear()
        stats.record_fingerprints = det
        with deterministic() if det else nullcontext():
            loss = compute_loss(ddp_model, batch)                           # 1. forward, backward
            loss.backward()
            nonfinite = nonfinite_ranks(stats)                              # 2. non-finite check
            pairs, table = [], None
            if audit or nonfinite:                                          # 3. ring replay
                pairs, table = ring_audit(module, compute_loss, rebuild_batch, step, stats,
                                          loss, sched.peer_offset, with_grads=det)
            if nonfinite or pairs:                                          # 4. gate
                optimizer.zero_grad(set_to_none=True)                       # step not applied
                culprits, ambiguous = localize(pairs)
                verdicts = [confirm(module, compute_loss, rebuild_batch, step, p, table, stats,
                                    sched.peer_offset, with_grads=det) for p in ambiguous]
                return {"step": step, "check": "compute", "nonfinite": nonfinite,
                        "mismatched_pairs": pairs, "culprits": culprits, "verdicts": verdicts,
                        "suspect_window": _window(last_audit, step, start_step),
                        "last_audit_pass": last_audit, "last_weight_check_pass": last_weights}
        if audit:
            last_audit = step
        grad_norm = torch.nn.utils.clip_grad_norm_(params, sched.max_grad_norm)
        bad = (~torch.isfinite(grad_norm)).to(torch.int32).reshape(1)        # reduced gradient
        dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=dp_group)
        if bad.item():
            optimizer.zero_grad(set_to_none=True)                           # step not applied
            return {"step": step, "check": "reduced gradient", "suspect_window": [step, step],
                    "last_audit_pass": last_audit, "last_weight_check_pass": last_weights}
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ckpt = step % sched.ckpt_every == 0
        if step % sched.check_every == 0 or (ckpt and sched.check_on_checkpoint):
            wc = find_divergent_ranks(params, dp_group)                     # 5. weight check
            if wc["diverged"]:
                return {"step": step, "check": "weights", "suspects": wc["suspects"],
                        "suspect_window": _window(last_weights, step, start_step),
                        "last_audit_pass": last_audit, "last_weight_check_pass": last_weights}
            last_weights = step
        if ckpt:                                                            # 6. checkpoint
            save_checkpoint(step, tags={"last_audit_pass": last_audit,
                                        "last_weight_check_pass": last_weights})
    return None
