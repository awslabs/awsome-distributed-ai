# GB200 parallelism reference (Megatron FP8/FP4)

Rule: keep **TP** (and **EP** for MoE) inside one 72-GPU NVLink domain so their collectives ride NVLink/NVSwitch (NVLS). Scale with **DP** and **PP** across EFA between UltraServers.

| Model | GPUs | TP | PP | DP | Domain layout | Notes |
|---|---|---|---|---|---|---|
| Llama3 8B | 8 | 4 | 1 | 2 | intra (1 UltraServer) | smoke/throughput; fits easily |
| Llama3 70B | 64 | 8 | 1 | 8 | intra (1 UltraServer, ≤72) | TP=8 stays on NVLink |
| Llama3 70B | 144 | 8 | 2 | 9 | 2 UltraServers | PP=2 crosses EFA; TP stays intra |
| Mixtral 8x7B (MoE) | 64 | 4 | 1 | — / EP=8 | intra | EP across the NVLink domain (NVLS all-to-all) — see the MoE/DeepEP sample |
| DeepSeek-V3-class (MoE) | 144+ | 8 | 2 | EP≤72 | 2+ UltraServers | EP capped at the 72-GPU domain; cross-domain rides EFA (no SHARP) |

All numbers are starting points to validate on hardware; FP8 (`fp8-mxfp8`) is the default precision on B200. `fp4-nvfp4` is eval/throughput only.
