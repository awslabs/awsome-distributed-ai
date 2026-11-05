#!/usr/bin/env python3
"""Tests sdc_order.py with software-injected faults on N nodes x 8 GPUs (N >= 3).

Writes, on rank 0, to $SDC_OUT (default /results):
  order.json     every result record: environment, topology, unit-style checks, timing,
                 fault scenarios, and healthy runs
  norms.json.gz  every rank's local gradient norm on every step of each scenario and healthy
                 run, for evaluating a gradient-norm outlier test offline
  rank_map.json  rank -> node, local rank, and GPU UUID (your real node names and GPU UUIDs)
"""
import gzip
import hashlib
import json
import os
import time
import traceback
from collections import Counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import model as M
import sdc_order as S
from common import (G, LOCAL_RANK, NNODES, NPN, RANK, SMOKE, WORLD, Data, Snapshot, _clean, cfg,
                    init, log, lr_at, now, sync)
from faults import Fault, adamw, autocast, ce

OUT = os.environ.get("SDC_OUT", "/results")
RECS = []
NORMS = {}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def add(kind, obj):
    if RANK == 0:
        rec = _clean({"kind": kind, "utc": utc(), **obj})
        RECS.append(rec)
        s = json.dumps(rec)
        print("RESULT " + (s if len(s) < 4000 else s[:4000] + " ...[truncated]"), flush=True)
        os.makedirs(OUT, exist_ok=True)
        with open(f"{OUT}/order.json", "w") as f:
            json.dump(RECS, f)


def flush_norms():
    if RANK == 0:
        with gzip.open(f"{OUT}/norms.json.gz", "wt") as f:
            json.dump(NORMS, f)


def compute_loss(model, batch):
    x, y = batch
    with autocast():
        logits = model(x)
    return ce(logits, y)


class Ctx:
    pass


X = Ctx()
X.faults = []


def rebuild_batch(step, rank):
    return X.data.seq(rank, step, X.c["B"], X.c["S"])


def local_norms():
    """Every rank's local gradient norm for the step that just ran (from the comm hook)."""
    local = torch.stack(X.stats.sq).sum().sqrt().reshape(1).float()
    out = torch.empty(WORLD, device="cuda")
    dist.all_gather_into_tensor(out, local)
    return [float(f"{v:.7g}") for v in out.tolist()]


class Loader:
    """Yields this rank's batches; sets the LR and the fault step. Instrumentation outside the
    loop: step times, the local weight fingerprints at the start of each step, and every rank's
    local gradient norm for each completed step."""

    def __init__(self, start, n, track=False, norms=None):
        self.start, self.n, self.track, self.norms = start, n, track, norms
        self.times, self.start_fp = [], {}
        self.last_t, self.last_s = None, None

    def _record_norms(self):
        if self.norms is not None and self.last_s is not None and X.stats.sq:
            v = local_norms()
            if RANK == 0:
                self.norms.append([self.last_s, v])

    def __iter__(self):
        for s in range(self.start, self.start + self.n):
            self._record_norms()
            sync()
            t = now()
            if self.last_t is not None:
                self.times.append((self.last_s, t - self.last_t))
            self.last_t, self.last_s = t, s
            for g in X.opt.param_groups:
                g["lr"] = lr_at(X.c, s)
            for f in X.faults:
                f.set_step(s)
            if self.track:
                self.start_fp[s] = S.fingerprints(X.params)
            yield rebuild_batch(s, RANK)

    def close_step(self):
        sync()
        if self.last_t is not None:
            self.times.append((self.last_s, now() - self.last_t))
        self._record_norms()


SAVES = []


def save_checkpoint(step, tags):
    """Test stand-in: writes nothing. Records the save decision, its tags, and whether the copy
    that would be saved (rank 0's) matches the per-tensor majority of all copies."""
    fp = S.fingerprints(X.params).flatten()
    out = torch.empty(WORLD * fp.numel(), dtype=torch.int64, device=fp.device)
    dist.all_gather_into_tensor(out, fp)
    tb = out.view(WORLD, -1).cpu()
    ref = tb.mode(dim=0).values
    SAVES.append({"step": step, "tags": tags,
                  "saved_copy_matches_majority": bool((tb[0] == ref).all())})


def gather_equal(t):
    """True when every rank's tensor t is identical (by value)."""
    t = t.flatten().contiguous()
    out = torch.empty(WORLD * t.numel(), dtype=t.dtype, device=t.device)
    dist.all_gather_into_tensor(out, t)
    out = out.view(WORLD, -1)
    return bool((out == out[0]).all())


def state_report(loader, alarm_step):
    """After train() returns: are every rank's weights and optimizer state finite, are they
    identical across ranks, and did the alarm step leave the weights unchanged?"""
    fin = torch.tensor([float(all(bool(torch.isfinite(p).all()) for p in X.params))], device="cuda")
    ost = [v for st in X.opt.state.values() for v in st.values()
           if torch.is_tensor(v) and v.is_cuda and v.is_floating_point()]
    ofin = torch.tensor([float(all(bool(torch.isfinite(v).all()) for v in ost))], device="cuda")
    unchanged = torch.tensor([-1.0], device="cuda")
    if alarm_step in loader.start_fp:
        unchanged[0] = float(torch.equal(S.fingerprints(X.params), loader.start_fp[alarm_step]))
    for t in (fin, ofin):
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
    dist.all_reduce(unchanged, op=dist.ReduceOp.MIN)
    return {"weights_finite_all_ranks": bool(fin.item()),
            "optimizer_state_finite_all_ranks": bool(ofin.item()),
            "weights_identical_across_ranks": gather_equal(S.fingerprints(X.params)),
            "optimizer_state_identical_across_ranks": gather_equal(S.fingerprints(ost)),
            "weights_unchanged_by_alarm_step":
                None if unchanged.item() < 0 else bool(unchanged.item())}


def run_train(start, n, sched, track=False, norms=None):
    loader = Loader(start, n, track, norms)
    t0 = now()
    alarm = S.train(X.ddp, X.stats, X.opt, loader, compute_loss, rebuild_batch, save_checkpoint,
                    sched, start_step=start)
    loader.close_step()
    return alarm, loader, now() - t0


def make_fault(spec, base):
    fs = dict(spec)
    path = fs.pop("path")
    start, stop = base + fs.pop("start"), base + fs.pop("stop")
    f = Fault(fs.pop("kind"), fs.pop("rank"), fs.pop("mode"), start, stop, **fs)
    mod = X.module.get_submodule(path)
    if f.kind in ("fwd", "dgrad"):
        f.install(module=mod)
    else:
        f.install(param=mod.weight)
    return f


def scenario(name, sched, faults=(), n=50, expect=""):
    X.snap.restore(X.params, X.opt)
    SAVES.clear()
    base = X.base
    X.faults = [make_fault(spec, base) for spec in faults]
    norms = NORMS.setdefault(name, [])
    alarm, loader, secs = run_train(base, n, sched, track=True, norms=norms)
    for f in X.faults:
        f.remove()
    metas = [f.meta() for f in X.faults]
    X.faults = []
    rep = state_report(loader, alarm["step"] if alarm else None)
    add("scenario", {"name": name, "expect": expect, "sched": vars(sched), "faults": metas,
                     "base_step": base, "alarm": alarm,
                     "alarm_rel_step": (alarm["step"] - base) if alarm else None,
                     "state_after": rep, "saves": list(SAVES), "steps_run": len(loader.times),
                     "seconds": secs})
    flush_norms()
    log(f"{name}: alarm={alarm and (alarm['step'] - base, alarm['check'])}")


def healthy(name, sched, minutes):
    X.snap.restore(X.params, X.opt)
    s = X.base
    deadline = now() + 60 * minutes
    steps, alarms, times = 0, [], []
    audits = checks = 0
    norms = NORMS.setdefault(name, [])
    while True:
        flag = torch.tensor([1 if (RANK == 0 and now() > deadline) else 0], device="cuda")
        dist.broadcast(flag, 0)
        if flag.item():
            break
        alarm, loader, _ = run_train(s, 200, sched, norms=norms)
        times += loader.times
        ran = 200 if alarm is None else alarm["step"] - s + 1
        audits += sum(1 for k in range(s, s + ran) if k % sched.audit_every == 0)
        checks += sum(1 for k in range(s, s + ran) if k % sched.check_every == 0)
        steps += ran
        s += ran
        if alarm is not None:
            alarms.append(alarm)
    ts = sorted(t for _, t in times)
    add("healthy", {"name": name, "sched": vars(sched), "minutes": minutes, "steps": steps,
                    "audit_steps": audits, "pair_comparisons": audits * WORLD,
                    "weight_checks": checks, "alarms": alarms, "n_alarms": len(alarms),
                    "median_step_ms": 1e3 * ts[len(ts) // 2] if ts else None})
    flush_norms()
    log(f"{name}: {steps} steps, {audits} audits, {checks} weight checks, {len(alarms)} alarms")


# ---------------------------------------------------------------------------------------------
# Unit-style checks that run on the cluster
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def plain_sum(tensors):
    """A single wrapping sum of raw bit patterns: the baseline the two-sum fingerprint improves on."""
    return torch.stack([t.contiguous().view(-1).view(torch.int32).to(torch.int64).sum()
                        for t in tensors])


def fingerprint_unit():
    """Which single-tensor changes each fingerprint sees (rank 0, local)."""
    if RANK != 0:
        return
    base = X.module.blocks[0].mlp.down.weight.detach().float().clone()
    cases = {}
    for case in ("one_bit_flip_bit0", "one_bit_flip_bit22", "swap_two_elements",
                 "compensate_two_elements", "two_elements_flipped"):
        t = base.clone()
        flat = t.view(-1)
        iv = flat.view(torch.int32)
        if case == "one_bit_flip_bit0":
            iv[123] ^= 1
        elif case == "one_bit_flip_bit22":
            iv[123] ^= 1 << 22
        elif case == "swap_two_elements":
            a = flat[123].clone()
            flat[123] = flat[124]
            flat[124] = a
        elif case == "compensate_two_elements":
            iv[123] += 4096
            iv[999] -= 4096
        elif case == "two_elements_flipped":
            iv[123] ^= 1 << 5
            iv[777] ^= 1 << 9
        f0, f1 = S.fingerprints([base]), S.fingerprints([t])
        cases[case] = {"elements_changed": int((t != base).sum()),
                       "plain_sum_detects": bool(plain_sum([base]) != plain_sum([t])),
                       "unweighted_detects": bool(f0[0, 0] != f1[0, 0]),
                       "fingerprints_detect": not torch.equal(f0, f1)}
    add("fingerprint_unit", {"tensor": "blocks.0.mlp.down.weight", "cases": cases})


def majority_unit():
    """find_divergent_ranks on small groups, with one GPU per node: a single bad copy, a tie,
    and no majority. Every rank creates the groups; members run the cases."""
    one_per_node = [k * NPN for k in range(NNODES)]
    groups = {"all_nodes": (one_per_node, dist.new_group(one_per_node)),
              "three_nodes": (one_per_node[:3], dist.new_group(one_per_node[:3]))}
    results = {}
    for gname, (members, grp) in groups.items():
        m = len(members)
        cases = {"one_bad_copy": {1: 1.0}}
        if m >= 4:
            cases["two_two_split"] = {2: 1.0, 3: 1.0}
            cases["no_majority"] = {2: 1.0, 3: 2.0}
        if m == 3:
            cases["three_distinct"] = {1: 1.0, 2: 2.0}
        for cname, change in cases.items():
            r = None
            if RANK in members:
                idx = members.index(RANK)
                t = torch.arange(4096, dtype=torch.float32, device="cuda") * 1e-3
                if idx in change:
                    t[7] += change[idx]
                r = S.find_divergent_ranks([t], grp)
            res = [None] * WORLD
            dist.all_gather_object(res, r)
            results[f"{gname}/{cname}"] = {"members": members,
                                           "changed": [members[i] for i in change],
                                           "result": res[members[0]]}
    add("majority_unit", {"cases": results})


def topology_unit():
    """check_topology on the real layout and on two invalid offsets, and where the confirming GPU
    would sit with confirmer = target + 2 * offset versus the one train() uses, target + offset."""
    nodes = S.check_topology(G["world"], NPN)
    res = {"nodes": len(set(nodes)), "ranks_per_node": dict(Counter(Counter(nodes).values()))}
    for bad in (NPN // 2, WORLD // 2):
        try:
            S.check_topology(G["world"], bad)
            res[f"offset_{bad}"] = "accepted"
        except ValueError as e:
            res[f"offset_{bad}"] = f"rejected: {e}"
    n, o = WORLD, NPN
    two_offsets_same = one_offset_same = 0
    for r in range(n):                  # r replays t = r + o
        t = (r + o) % n
        two_offsets_same += nodes[(t + 2 * o) % n] in (nodes[r], nodes[t])
        one_offset_same += nodes[(t + o) % n] in (nodes[r], nodes[t])
    res["pairs"] = n
    res["pairs_where_confirmer_target_plus_2_offsets_shares_a_node"] = two_offsets_same
    res["pairs_where_confirmer_target_plus_offset_shares_a_node"] = one_offset_same
    add("topology_unit", res)


def rank_map():
    info = {"rank": RANK, "local_rank": LOCAL_RANK, "node": os.environ.get("NODE_NAME"),
            "cuda_device": torch.cuda.current_device(),
            "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)}
    table = [None] * WORLD
    dist.all_gather_object(table, info)
    if RANK == 0:
        with open(f"{OUT}/rank_map.json", "w") as f:
            json.dump(table, f, indent=1)
    add("rank_map_written", {"ranks": WORLD, "nodes": len({t["node"] for t in table})})


# ---------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------
def median_ms(fn, warm=3, reps=20):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        sync()
        t0 = now()
        fn()
        sync()
        ts.append(now() - t0)
    ts.sort()
    return 1e3 * ts[len(ts) // 2]


def timing():
    res = {}
    m = M.Llama(X.c).cuda()
    M.init_full(m, X.c)
    ddp = DDP(m, device_ids=[LOCAL_RANK], bucket_cap_mb=25)
    opt = adamw(list(ddp.parameters()), X.c)
    for label, det in (("plain_ddp_ms", False), ("deterministic_only_ms", True)):
        ts = []
        for s in range(35):
            sync()
            t0 = now()
            with S.deterministic() if det else torch.enable_grad():
                loss = compute_loss(ddp, rebuild_batch(s, RANK))
                loss.backward()
            torch.nn.utils.clip_grad_norm_(list(ddp.parameters()), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sync()
            ts.append(now() - t0)
        ts = sorted(ts[5:])
        res[label] = 1e3 * ts[len(ts) // 2]
    del ddp, opt, m
    torch.cuda.empty_cache()
    dist.barrier()
    big = 10 ** 9
    X.snap.restore(X.params, X.opt)
    _, ld, _ = run_train(X.base, 40, S.Schedule(audit_every=big, check_every=big, ckpt_every=big,
                                                peer_offset=NPN))
    t = sorted(x for _, x in ld.times[5:])
    res["loop_nonaudit_ms"] = 1e3 * t[len(t) // 2]
    for det in (False, True):
        X.snap.restore(X.params, X.opt)
        _, ld, _ = run_train(X.base, 30, S.Schedule(audit_every=1, deterministic_audits=det,
                                                    check_every=big, ckpt_every=big,
                                                    peer_offset=NPN))
        t = sorted(x for _, x in ld.times[3:])
        res[f"loop_audit_{'det' if det else 'default'}_ms"] = 1e3 * t[len(t) // 2]
    X.snap.restore(X.params, X.opt)
    res["fingerprints_local_ms"] = median_ms(lambda: S.fingerprints(X.params))
    res["plain_sum_local_ms"] = median_ms(lambda: plain_sum(X.params))
    res["weight_check_ms"] = median_ms(lambda: S.find_divergent_ranks(X.params, G["world"]))
    res["method"] = ("step times: medians over 30 (plain, deterministic-only), 35 (non-audit) and "
                     "27 (audit) steps after warm-up; fingerprint and weight-check times: medians "
                     "of 20 calls after 3 warm-up calls")
    add("timing", res)


def sources():
    d = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".py"):
            with open(os.path.join(d, fn), "rb") as f:
                out[fn] = hashlib.sha256(f.read()).hexdigest()
    return out


def main():
    init()
    X.c = c = cfg()
    X.data = Data(c)
    add("env", {"torch": torch.__version__, "cuda": torch.version.cuda,
                "nccl": ".".join(map(str, torch.cuda.nccl.version())), "world": WORLD,
                "nodes": NNODES, "gpus_per_node": NPN, "smoke": SMOKE, "cfg": c,
                "gpu": torch.cuda.get_device_name(), "sources_sha256": sources()})
    rank_map()
    m = M.Llama(c).cuda()
    M.init_full(m, c)
    X.module = m
    X.ddp = DDP(m, device_ids=[LOCAL_RANK], bucket_cap_mb=25)
    X.stats = S.attach(X.ddp, G["world"])
    X.params = list(X.ddp.parameters())
    X.opt = adamw(X.params, c)
    orig_step = X.opt.step

    def stepped(*a, **k):
        r = orig_step(*a, **k)
        for f in X.faults:
            f.after_step()
        return r
    X.opt.step = stepped

    big = 10 ** 9
    W = ((c["W"] + 19) // 20) * 20          # align scenario steps with the audit/check schedule
    t0 = now()
    alarm, _, _ = run_train(0, W, S.Schedule(audit_every=big, check_every=50, ckpt_every=big,
                                             peer_offset=NPN))
    log(f"warmup {W} steps in {now() - t0:.1f}s, alarm={alarm}")
    X.base = W
    X.snap = Snapshot(X.params, X.opt)
    sync()

    topology_unit()
    fingerprint_unit()
    majority_unit()
    timing()

    std = dict(audit_every=10, deterministic_audits=True, check_every=10, ckpt_every=20,
               peer_offset=NPN)
    dflt = {**std, "deterministic_audits": False}
    down = f"blocks.{c['L'] // 2 - 1}.mlp.down"

    def fwd(rank, mode, value=None, start=23, stop=60, once=False, path=down, kind="fwd"):
        return dict(kind=kind, rank=rank, mode=mode, value=value, start=start, stop=stop,
                    path=path, once_per_step=once)

    scenario("healthy_reference", S.Schedule(**std))
    scenario("nan_fwd_r13_once_nonaudit", S.Schedule(**std),
             [fwd(13, "nan", start=25, stop=26, once=True)],
             expect="check 1 at step 25; step not applied")
    scenario("fwd_x1000_r13_persistent", S.Schedule(**std), [fwd(13, "scale", 1000.0)],
             expect="missed on steps 23-29; audit at 30 names 13")
    scenario("fwd_x1000_r13_transient_at_audit", S.Schedule(**std),
             [fwd(13, "scale", 1000.0, start=30, stop=31, once=True)],
             expect="one pair at 30; confirmation: transient on 13")
    scenario("fwd_x1000_r5_persistent_faulty_replayer", S.Schedule(**std), [fwd(5, "scale", 1000.0)],
             expect="audit at 30 names 5, not 13")
    scenario("fwd_x10000_r13_persistent", S.Schedule(**std), [fwd(13, "scale", 10000.0)],
             expect="audit at 30 names 13")
    scenario("fwd_noise10pct_r13_persistent", S.Schedule(**std), [fwd(13, "noise", 0.1)],
             expect="audit at 30 names 13")
    scenario("dgrad_x1000_r13_persistent_det", S.Schedule(**std),
             [fwd(13, "scale", 1000.0, kind="dgrad")],
             expect="audit at 30 names 13 (input gradient, backward only)")
    scenario("wgrad_x1000_r13_persistent_det", S.Schedule(**std),
             [fwd(13, "scale", 1000.0, kind="wgrad")],
             expect="losses match, gradient fingerprints differ; names 13")
    scenario("wgrad_x1000_r13_persistent_default", S.Schedule(**dflt),
             [fwd(13, "scale", 1000.0, kind="wgrad")],
             expect="loss-only audits can't see it: no alarm")
    for tag, sch in (("det", std), ("default", dflt)):
        scenario(f"fwd_bf16bit3_r13_persistent_{tag}", S.Schedule(**sch), [fwd(13, "bitflip", 3)],
                 expect="audit at 30 names 13")
        scenario(f"fwd_lane_x1.1_r13_persistent_{tag}", S.Schedule(**sch), [fwd(13, "lane", 1.1)],
                 expect="audit at 30 names 13")
    scenario("two_faulty_ranks_fwd_x1000_r13_r22", S.Schedule(**std),
             [fwd(13, "scale", 1000.0), fwd(22, "scale", 1000.0)],
             expect="audit at 30 names 13 and 22")
    ck = dict(std, ckpt_every=5)
    wfault = dict(kind="weight", rank=0, mode="bitflip", value=22, start=31, stop=32, path=down)
    scenario("weight_bit22_r0_check_on_ckpt", S.Schedule(**ck), [wfault],
             expect="forced check at checkpoint step 35 names 0; no save at 35")
    scenario("weight_bit22_r0_no_check_on_ckpt", S.Schedule(**{**ck, "check_on_checkpoint": False}),
             [wfault], expect="save at 35 from the corrupted copy; audit at 40 names 0")
    wonly = dict(audit_every=big, deterministic_audits=True, check_every=10, ckpt_every=20,
                 peer_offset=NPN)
    for mode, value in (("swap", None), ("compensate", 4096)):
        scenario(f"weight_{mode}_r0_weight_check_only", S.Schedule(**wonly),
                 [dict(wfault, mode=mode, value=value)],
                 expect="weight check at 40 names 0 (the plain bit sum is unchanged)")

    X.snap.restore(X.params, X.opt)       # verify on resume: a corrupted load on rank 7
    if RANK == 7:
        with torch.no_grad():
            w = X.module.blocks[3].attn.q.weight.view(-1)
            w.view(torch.int32)[4242] ^= 1 << 22
    add("resume_check", {"name": "corrupted_load_r7",
                         "result": S.find_divergent_ranks(X.params, G["world"])})

    det_min = float(os.environ.get("SDC_HEALTHY_MIN", "0.5" if SMOKE else "45"))
    healthy("healthy_det_audit_every_step",
            S.Schedule(audit_every=1, deterministic_audits=True, check_every=10, ckpt_every=big,
                       peer_offset=NPN), det_min)
    healthy("healthy_default_audit_every_step",
            S.Schedule(audit_every=1, deterministic_audits=False, check_every=10, ckpt_every=big,
                       peer_offset=NPN), det_min / 2)
    healthy("healthy_production_schedule",
            S.Schedule(audit_every=10, deterministic_audits=True, check_every=10, ckpt_every=big,
                       peer_offset=NPN), det_min)
    add("done", {})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(f"RANK {RANK} FAILED:\n{traceback.format_exc()}", flush=True)
        raise
