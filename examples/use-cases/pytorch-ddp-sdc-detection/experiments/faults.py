"""Fault injection and training helpers for the experiments in run_order.py.

Faults are injected in software, on a chosen rank and step:
  fwd    - change a layer's forward output value, keep its gradient path
  dgrad  - change the gradient flowing into a layer's input (backward only)
  wgrad  - change a layer's weight gradient (backward only)
  weight - change one rank's stored weight right after optimizer.step()

Modes:
  scale      - multiply one element by `value`
  nan        - set one element to NaN
  bitflip    - flip bit `value` of one element's bit pattern
  noise      - multiply every element by (1 + value * standard normal noise)
  lane       - multiply one column (the last dimension) by `value`
  swap       - swap two elements (the plain bit sum doesn't change)
  compensate - add +value and -value to two elements' bit patterns (same plain bit sum)
"""
import torch
import torch.nn.functional as F

from common import RANK

_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _xor_mask(bit, esize):
    m = 1 << bit
    if bit == 8 * esize - 1:
        m -= 1 << (8 * esize)
    return m


class Fault:
    """kind: fwd | dgrad | wgrad | weight. mode: see the module docstring."""

    def __init__(self, kind, rank, mode, start, stop=None, value=None, elem=123_457, lane=1234,
                 once_per_step=False):
        self.kind, self.rank, self.mode = kind, rank, mode
        self.start = start
        self.stop = stop if stop is not None else start + 1
        self.value, self.elem, self.lane = value, elem, lane
        self.once_per_step = once_per_step
        self.t = -1
        self.applied_at = None
        self.handles = []
        self.module = None
        self.param = None

    def meta(self):
        return {k: getattr(self, k) for k in ("kind", "rank", "mode", "start", "stop", "value",
                                              "elem", "lane", "once_per_step")}

    def set_step(self, t):
        self.t = t

    def active(self):
        if RANK != self.rank or not (self.start <= self.t < self.stop):
            return False
        if self.once_per_step and self.applied_at == self.t:
            return False
        return True

    def _mark(self):
        self.applied_at = self.t

    @torch.no_grad()
    def apply_(self, out):
        m = self.mode
        if m == "lane":
            out[..., self.lane % out.shape[-1]] *= self.value
            return
        if m == "noise":
            out.mul_(1 + self.value * torch.randn_like(out))
            return
        flat = out.view(-1)
        e = self.elem % flat.numel()
        if m == "scale":
            flat[e] = flat[e] * self.value
        elif m == "nan":
            flat[e] = float("nan")
        elif m == "bitflip":
            iv = flat.view(_INT_VIEW[out.element_size()])
            iv[e] ^= _xor_mask(int(self.value), out.element_size())
        elif m == "swap":         # two elements trade places: the plain bit sum doesn't change
            j = (e + 1) % flat.numel()
            a = flat[e].clone()
            flat[e] = flat[j]
            flat[j] = a
        elif m == "compensate":   # +value and -value on two bit patterns: same plain bit sum
            iv = flat.view(_INT_VIEW[out.element_size()])
            j = (e + 1) % flat.numel()
            iv[e] += int(self.value)
            iv[j] -= int(self.value)

    def _corrupt(self, t):
        o = t.detach().clone()
        self.apply_(o)
        self._mark()
        return o

    def _corrupt_fwd(self, t):
        """Wrong forward value, unchanged backward: c + (t - t.detach()) has the bits of c and
        the gradient of t (the backward GEMMs don't know the forward result was wrong)."""
        c = t.detach().clone()
        self.apply_(c)
        self._mark()
        return c + (t - t.detach())

    def install(self, module=None, param=None):
        self.module, self.param = module, param
        k = self.kind
        if k == "fwd":
            def fh(mod, inp, out):
                return self._corrupt_fwd(out) if self.active() else None
            self.handles.append(module.register_forward_hook(fh))
        elif k == "dgrad":
            def pre(mod, inp):
                x = inp[0]
                if x.requires_grad and RANK == self.rank:
                    x.register_hook(lambda g: self._corrupt(g) if self.active() else None)
            self.handles.append(module.register_forward_pre_hook(pre))
        elif k == "wgrad":
            self.handles.append(param.register_hook(
                lambda g: self._corrupt(g) if self.active() else None))
        return self

    @torch.no_grad()
    def after_step(self):
        if self.kind == "weight" and RANK == self.rank and self.t == self.start:
            self.apply_(self.param.data)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def ce(logits, y):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))


def adamw(params, c):
    return torch.optim.AdamW(params, lr=c["LR"], betas=(0.9, 0.95), weight_decay=0.1)
