"""Llama-style decoder used by the experiments (RMSNorm, RoPE, SwiGLU MLP, causal attention).
Plain PyTorch only."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


def rope_cache(S, hd, device):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    t = torch.arange(S, device=device, dtype=torch.float32)
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    # x [B, h, S, hd]; cos/sin [S, hd/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], -1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.hl, self.hd = H, D // H
        w = self.hl * self.hd
        self.q = nn.Linear(D, w, bias=False)
        self.k = nn.Linear(D, w, bias=False)
        self.v = nn.Linear(D, w, bias=False)
        self.o = nn.Linear(w, D, bias=False)

    def forward(self, x, rope):
        B, S, _ = x.shape
        q = self.q(x).view(B, S, self.hl, self.hd).transpose(1, 2)
        k = self.k(x).view(B, S, self.hl, self.hd).transpose(1, 2)
        v = self.v(x).view(B, S, self.hl, self.hd).transpose(1, 2)
        cos, sin = rope
        q = apply_rope(q, cos[:S], sin[:S])
        k = apply_rope(k, cos[:S], sin[:S])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, S, self.hl * self.hd))


class MLP(nn.Module):
    def __init__(self, D, F_):
        super().__init__()
        self.gate = nn.Linear(D, F_, bias=False)
        self.up = nn.Linear(D, F_, bias=False)
        self.down = nn.Linear(F_, D, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = RMSNorm(c["D"])
        self.attn = Attn(c["D"], c["H"])
        self.n2 = RMSNorm(c["D"])
        self.mlp = MLP(c["D"], c["FFN"])

    def forward(self, x, rope):
        x = x + self.attn(self.n1(x), rope)
        return x + self.mlp(self.n2(x))


class Llama(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c["V"], c["D"])
        self.blocks = nn.ModuleList([Block(c) for _ in range(c["L"])])
        self.norm = RMSNorm(c["D"])
        self.head = nn.Linear(c["D"], c["V"], bias=False)
        cos, sin = rope_cache(c["S"], c["D"] // c["H"], "cuda")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h, (self.cos, self.sin))
        return self.head(self.norm(h))


@torch.no_grad()
def init_full(model, c, seed=1234):
    """Deterministic init (the same on every rank)."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    std = 0.02
    for name, p in model.named_parameters():
        if p.dim() == 1:
            p.fill_(1.0)
        elif name.endswith(("attn.o.weight", "mlp.down.weight")):
            p.normal_(0.0, std / math.sqrt(2 * c["L"]), generator=g)
        else:
            p.normal_(0.0, std, generator=g)
