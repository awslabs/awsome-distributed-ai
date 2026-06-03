# torchtitan Training on AWS

[torchtitan](https://github.com/pytorch/torchtitan) is a reference architecture for large-scale LLM training using native PyTorch. It aims to showcase PyTorch's latest distributed training features in a clean, minimal code base. The library is designed to be simple to understand, use, and extend for different training purposes, with minimal changes required to the model code when applying various parallel processing techniques.

## Key Features

torchtitan offers several advanced capabilities:

- [FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md) with per-parameter sharding
- [FP8 Support](https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323)
- [Async Tensor Parallelism](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487) in PyTorch
- [Optimized Checkpointing Efficiency](https://discuss.pytorch.org/t/distributed-w-torchtitan-optimizing-checkpointing-efficiency-with-pytorch-dcp/211250) with PyTorch DCP
- [Zero-Bubble Pipeline Parallelism](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
- [Context Parallelism for training long context LLMs](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082/1) (with 1M Sequence Length)