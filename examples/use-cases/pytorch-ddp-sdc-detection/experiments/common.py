"""Shared setup for the experiments: process groups, configuration, synthetic data, snapshots.

SDC_SMOKE=1 selects a small model for a quick end-to-end check of the job wiring."""
import copy
import math
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
NPN = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
NNODES = WORLD // NPN
SMOKE = os.environ.get("SDC_SMOKE", "0") == "1"


def cfg():
    if SMOKE:
        return dict(V=4096, D=256, L=4, H=8, FFN=704, S=256, B=2, W=6, LR=3e-4, WARM=3, COST=20)
    return dict(V=32000, D=2048, L=16, H=16, FFN=5632, S=2048, B=2, W=400, LR=3e-4, WARM=100, COST=1000)


def lr_at(c, step):
    return c["LR"] * min(1.0, (step + 1) / c["WARM"])


G = {}


def init():
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group("nccl", timeout=timedelta(minutes=int(os.environ.get("SDC_TIMEOUT_MIN", "30"))),
                            device_id=torch.device("cuda", LOCAL_RANK))
    G["world"] = dist.group.WORLD


def sync():
    torch.cuda.synchronize()


def now():
    return time.perf_counter()


def _clean(o):
    if isinstance(o, float):
        if math.isnan(o):
            return "nan"
        if math.isinf(o):
            return "inf" if o > 0 else "-inf"
        return float(f"{o:.6g}")
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, set):
        return sorted(o)
    return o


def log(msg):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Data:
    """Synthetic copy task: Zipf tokens, first half repeated, so attention has something to learn."""

    def __init__(self, c):
        self.c = c
        w = 1.0 / torch.arange(1, c["V"] + 1, device="cuda", dtype=torch.float32) ** 1.1
        self.p = w / w.sum()

    def seq(self, index, step, B, S):
        g = torch.Generator(device="cuda")
        g.manual_seed(1_000_003 * step + 7919 * index + 17)
        half = S // 2 + 1
        first = torch.multinomial(self.p.unsqueeze(0).expand(B, -1).contiguous(), half,
                                  replacement=True, generator=g)
        s = torch.cat([first, first], 1)            # [B, 2*half] = [B, S+2]
        return s[:, :S].contiguous(), s[:, 1:S + 1].contiguous()


def _cpu(o):
    if torch.is_tensor(o):
        return o.detach().to("cpu", copy=True)
    if isinstance(o, dict):
        return {k: _cpu(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_cpu(v) for v in o]
    if isinstance(o, tuple):
        return tuple(_cpu(v) for v in o)
    return copy.deepcopy(o)


class Snapshot:
    """CPU copy of parameters and optimizer state, restored before each scenario."""

    def __init__(self, params, opt):
        self.p = [p.detach().to("cpu", copy=True) for p in params]
        self.o = _cpu(opt.state_dict())

    def restore(self, params, opt):
        with torch.no_grad():
            for p, s in zip(params, self.p):
                p.copy_(s)
        st = {"state": {k: {kk: (vv.clone() if torch.is_tensor(vv) and kk == "step" else vv)
                            for kk, vv in v.items()} for k, v in self.o["state"].items()},
              "param_groups": copy.deepcopy(self.o["param_groups"])}
        opt.load_state_dict(st)
        for p in params:
            p.grad = None
