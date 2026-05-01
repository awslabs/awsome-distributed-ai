# Experimental Scripts

Scripts in this directory are experimental and not recommended for production use.

## FSDP Launcher (`run_train_fsdp.py`)

An alternative training launcher that replaces DDP with FSDP (SHARD_GRAD_OP / ZeRO-2).
This approach was benchmarked and found to be approximately **2x slower** than the
standard DDP-based `run_train.py` launcher due to FSDP communication overhead on the
V-JEPA architecture.

**Recommendation:** Use the standard `run_train.py` with DDP for production training.
This FSDP launcher is preserved for reference and future experimentation only.
