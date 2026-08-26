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

The loop deliberately runs without megatron's DDP wrapper (to stay independent
of its DDP config API) but reproduces the two things DDP does that this gate
needs, restricted to the REPLICATED (non-expert) params:
  - at construction, broadcast them from rank 0 — megatron inits the router
    weight and position embeddings under a per-rank-varying RNG context
    (measured ~1.5e-1 cross-rank divergence at step 0);
  - each step, all-reduce (average) their gradients — with the same batch this
    is NOT a no-op for replicated params, because each rank backprops through
    DIFFERENT local experts, so the shared router receives different per-rank
    gradients and would re-diverge every step (measured spread 5.7e-4 -> 8.9e-3
    over 3 steps without the sync).
Expert params are deliberately left per-rank distinct at init and their
gradients are NOT reduced — that asymmetry is expert parallelism itself. With
replicated params in lock-step and the same batch, gate (2) cross-rank loss
agreement is a true transport test at any step count: a rank whose dispatched
tokens were corrupted in the MoE all-to-all disagrees, a correct one does not.
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
        # Dropout OFF. This is a transport-DETERMINISM gate: gate 2 asserts every rank
        # computes the SAME loss from the SAME seeded batch after the MoE all-to-all
        # round-trip. hidden/attention dropout default to 0.1 and inject per-rank-random
        # masks (the divergence compounds through weight updates: measured cross-rank
        # spread grew 0.07 -> 0.15 -> 0.47 over 3 steps with dropout on), which defeats
        # that invariant without indicating any transport fault. moe_input_jitter and
        # aux-loss are already off by default, so with dropout off the router decision and
        # expert compute are deterministic and a correct all-to-all yields bit-consistent
        # cross-rank loss (modulo FP-reduction noise, well under the 1e-2 gate).
        hidden_dropout=0.0,
        attention_dropout=0.0,
        # fp32 throughout — this is a transport-correctness gate, not a fidelity run.
        # The megatron `local` layer spec (get_gpt_layer_local_spec, chosen to keep this
        # driver free of megatron's TE/DDP wrapper machinery) emits fp32 LayerNorm
        # activations; with bf16 params the router's te_general_gemm then sees a
        # bf16-weight x fp32-input GEMM, which this NGC base's TransformerEngine cuBLASLt
        # build rejects ("unsupported value or parameter") regardless of moe_router_dtype
        # (measured: moe_utils RouterGatingLinearFunction always takes the TE path for any
        # router_dtype != fp64). fp32 params make every GEMM (fp32,fp32,fp32) — proven to
        # run fwd+bwd+opt here — and NCCL/DeepEP all-to-all is dtype-agnostic (DeepEP even
        # requires fp32 probs), so fp32 exercises the same transport path a bf16 run would
        # while making the cross-rank loss-agreement gate exact (no bf16 rounding near the
        # 1e-2 threshold). A bf16 run would instead need the TE layer spec, pulling in the
        # megatron wrapper machinery this driver deliberately avoids.
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
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

    # Sync REPLICATED params from rank 0 (what DDP does at construction; this driver
    # is deliberately DDP-free — see docstring). Megatron's default RNG tracker seeds
    # most replicated params identically, but the router weight and position embeddings
    # init under a per-rank-varying RNG context — MEASURED cross-rank divergence 1.5e-1
    # (router.weight) / 1.6e-1 (position_embeddings) at step 0, before any update. Left
    # unsynced, ranks route the same batch differently and gate 2 false-fails on a fully
    # correct image + transport. Broadcast makes gate 2's invariant hold AND keeps it a
    # real transport test (a corrupted all-to-all still diverges the combined output ->
    # loss). Expert params (.experts./.local_experts.) are correctly left per-rank
    # distinct — that asymmetry IS expert parallelism.
    for name, p in model.named_parameters():
        if ".experts." not in name and ".local_experts." not in name:
            dist.broadcast(p.data, src=0)

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

    # Partition params once: replicated (kept in lock-step across ranks, DDP-style) vs
    # expert (per-rank distinct — expert parallelism). Same split as the init broadcast.
    replicated_params = [p for n, p in model.named_parameters()
                         if ".experts." not in n and ".local_experts." not in n]

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    losses = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        # fp32 model (see config) — no autocast needed; every linear sees fp32 x fp32.
        per_token_loss = model(tokens, position_ids, attention_mask, labels=labels)
        loss = per_token_loss.float().mean()
        loss.backward()
        # All-reduce (average) gradients of REPLICATED params — the other half of what
        # DDP does. The original "same batch => grad all-reduce is a no-op" premise is
        # FALSE for these: each rank backprops through DIFFERENT local experts, so the
        # shared router/embeddings receive different gradients per rank. Left unsynced the
        # router re-diverges after every step (MEASURED: cross-rank spread grew
        # 5.7e-4 -> 8.9e-3 over 3 steps without this). Syncing keeps replicated params
        # bit-identical across ranks at every step, so gate 2 stays a tight transport test
        # regardless of TRAIN_STEPS, and the loss decrease is a true synchronized-train
        # decrease. Expert grads are deliberately NOT reduced — that is expert parallelism.
        for p in replicated_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
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
