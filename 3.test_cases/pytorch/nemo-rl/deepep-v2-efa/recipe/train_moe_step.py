#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""train_moe_step.py — N-step Megatron-core MoE training gate over EFA.

The training analogue of the rollout probe: builds a small expert-parallel GPT
MoE with megatron.core (random weights, synthetic batch — no checkpoint, no
dataset), runs N optimizer steps across all ranks, and gates on:

  1. loss finite at every step and lower at the last step than the first
     (the Wave-28 "Shape Y" oracle shape: loss 26.41 -> 24.62 over 3 steps on
     the measured substrate — exact values are seed/substrate-specific and NOT
     asserted, only finiteness + decrease are);
  2. cross-rank loss agreement — every rank feeds the SAME seeded batch, so
     after the MoE all-to-all round-trip all ranks must compute the SAME loss;
     a rank whose dispatched tokens were corrupted in flight disagrees here;
  3. a MIN all-reduced verdict — one bad rank fails every rank.

Dispatcher selection (MOE_DISPATCHER):
  alltoall  (default) Megatron's stock all-to-all dispatcher — NCCL all-to-all
            over EFA, no deep_ep involvement. This is the BASELINE gate: it
            runs on the upstream-only image.
  flex      Megatron's flex dispatcher with moe_enable_deepep=True — the
            DeepEP V2 ElasticBuffer path. Needs the opt-in draft-PR image
            (Megatron-LM#4632); train-step.sh refuses it on an unpatched
            image rather than failing with a distant import error.

Every rank sees the same synthetic batch, so data-parallel gradient
all-reduce would be a mathematical no-op — the loop deliberately runs without
a DDP wrapper to stay independent of megatron's DDP config API, and the
cross-rank loss-agreement gate (2) is what makes that sound.
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
LOCAL = int(os.environ["LOCAL_RANK"])

DISPATCHER = os.environ.get("MOE_DISPATCHER", "alltoall")
STEPS = int(os.environ.get("TRAIN_STEPS", "3"))
E = int(os.environ.get("EP_EXPERTS", "128"))
K = int(os.environ.get("EP_TOPK", "8"))
H = int(os.environ.get("EP_HIDDEN", "2048"))
SEQ = int(os.environ.get("TRAIN_SEQ", "256"))
MBS = int(os.environ.get("TRAIN_MBS", "2"))
VOCAB = int(os.environ.get("TRAIN_VOCAB", "8192"))
LR = float(os.environ.get("TRAIN_LR", "1e-3"))


def log(msg):
    print(f"[rank{RANK}] {msg}", flush=True)


def main():
    torch.cuda.set_device(LOCAL)
    dist.init_process_group("nccl", rank=RANK, world_size=WORLD)
    dev = torch.device("cuda", LOCAL)

    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    ep_size = int(os.environ.get("EP_SIZE", str(WORLD)))
    assert WORLD % ep_size == 0, f"world ({WORLD}) must divide by EP size ({ep_size})"
    assert E % ep_size == 0, f"experts ({E}) must divide by EP size ({ep_size})"

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
    )
    model_parallel_cuda_manual_seed(123)
    log(f"train-step start world={WORLD} ep={ep_size} dispatcher={DISPATCHER} "
        f"experts={E} topk={K} hidden={H} steps={STEPS}")

    config = TransformerConfig(
        num_layers=2,
        hidden_size=H,
        num_attention_heads=16,
        ffn_hidden_size=4 * H,
        moe_ffn_hidden_size=768,          # Qwen3-30B-A3B expert width
        num_moe_experts=E,
        moe_router_topk=K,
        moe_token_dispatcher_type=DISPATCHER,
        moe_enable_deepep=(DISPATCHER == "flex"),
        expert_model_parallel_size=ep_size,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(num_experts=E, moe_grouped_gemm=False),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model up: {n_params/1e6:.1f}M params on this rank")

    # Same seeded batch on every rank — see the module docstring for why that
    # makes the DDP-less loop sound and turns loss agreement into a gate.
    g = torch.Generator(device="cpu").manual_seed(1234)
    tokens = torch.randint(0, VOCAB, (MBS, SEQ), generator=g).to(dev)
    position_ids = torch.arange(SEQ, device=dev).unsqueeze(0).expand(MBS, -1)
    # boolean mask, True = masked (megatron local-spec attention convention)
    attention_mask = torch.triu(
        torch.ones(SEQ, SEQ, dtype=torch.bool, device=dev), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    labels = torch.roll(tokens, shifts=-1, dims=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    losses = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        per_token_loss = model(tokens, position_ids, attention_mask, labels=labels)
        loss = per_token_loss.float().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # gate 2: cross-rank agreement (same batch => same loss on every rank)
        t = torch.tensor([loss.item()], device=dev)
        t_min, t_max = t.clone(), t.clone()
        dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
        spread = (t_max - t_min).item()
        log(f"TRAIN-STEP step={step} loss={loss.item():.4f} cross-rank-spread={spread:.2e}")

    finite = all(l == l and abs(l) != float("inf") for l in losses)
    decreasing = losses[-1] < losses[0]
    agree = spread < 1e-2
    ok = finite and decreasing and agree
    log(f"TRAIN-STEP losses={['%.4f' % l for l in losses]} "
        f"finite={finite} decreasing={decreasing} cross-rank-agree={agree}")

    flag = torch.tensor([1 if ok else 0], device=dev, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    verdict = "TRAIN-STEP-PASS" if int(flag.item()) == 1 else "TRAIN-STEP-FAIL"
    log(f"{verdict} dispatcher={DISPATCHER} world={WORLD} ep={ep_size}")

    dist.barrier()
    parallel_state.destroy_model_parallel()
    return 0 if int(flag.item()) == 1 else 4


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print(f"[rank{RANK}] TRAIN-STEP-EXC", flush=True)
        sys.exit(1)
