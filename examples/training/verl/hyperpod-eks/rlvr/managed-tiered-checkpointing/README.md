# MTC GRPO Training with RayCluster

This directory contains configurations for running GRPO training with VERL with [HyperPod Managed Tiered Checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/managed-tier-checkpointing.html).

## Files

- `mtc-grpo-cluster.yaml` - RayCluster configuration
- `submit-mtc-grpo.sh` - Script to submit the GRPO training job to the Ray cluster
- `Dockerfile` - Dockerfile for building MTC-enabled training image
- `build-push.sh` - Script to build and push MTC-enabled Docker image

## Setup

0. Enabling MTC on Your HyperPod Cluster

Before using MTC, ensure your SageMaker HyperPod cluster has Managed Tiered Checkpointing enabled. Follow the [AWS documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/managed-tier-checkpointing-setup.html) to:

- Enable MTC on your cluster during creation or update
- Configure the memory allocation percentage (20-100%)


1. Source environment variables:
```bash
# 1. Load environment variables
source setup/env_vars
```

2. Create Service Account for your pods to have S3 access. To do this, please read the [IRSA-README.md](../setup/IRSA-README.md). 

## Build MTC-Enabled Docker Image

Before deploying the cluster, you need to build a Docker image with the MTC library installed:

```bash
# Navigate to the managed-tiered-checkpointing directory
cd managed-tiered-checkpointing

# Build and push the image
./build-push.sh
```

This will create an image tagged as `${REGISTRY}${IMAGE}:${TAG}-mtc` with the MTC dependencies installed.

## Deploy the RayCluster

```bash
envsubst < managed-tiered-checkpointing/mtc-grpo-cluster.yaml | kubectl apply -f -
```

## Clone MTC-enabled VERL Code

Delete existing verl repo if you already cloned:
```
rm -rf verl
```

Clone MTC-enabled VERL code. This is a fork from the main VERL repo that has modified checkpointing code to enabled managed tiered checkpointing:

```
git clone https://github.com/aruncs2005/verl.git
```

**⚠️ Note**: This fork is based on an older version of the main verl repository. It provides working out-of-the-box MTC support but may lack recent features and bug fixes from the main repository. For instructions on enabling MTC in the main verl repository yourself, see [How to Enable MTC in Main VERL](#how-to-enable-mtc-in-main-verl) below.

## Submit the training job

```bash
./managed-tiered-checkpointing/submit-mtc-grpo.sh
```

## Monitoring

- **Ray Dashboard**: http://localhost:8265 (after port forwarding)
- **View logs**: `kubectl logs -f <head-pod-name>`
- **Check job status**: `ray job status <job-id>`
- **Follow job logs**: `ray job logs <job-id> --follow`

## Configuration

Edit `submit-mtc-grpo.sh` to modify training parameters:

- `train_prompt_bsz` - Training batch size
- `train_prompt_mini_bsz` - Mini batch size for PPO
- `train_prompt_micro_bsz_per_gpu` - Micro batch size per GPU
- `n_resp_per_prompt` - Number of responses per prompt
- `gen_tp` - Tensor parallelism for generation
- Model path, data paths, S3 checkpoint location, etc.

**MTC-Specific Parameters**:
- `actor_rollout_ref.actor.checkpoint.s3_base_path` - S3 path for checkpoint storage
- `actor_rollout_ref.actor.checkpoint.ckpt_namespace` - Unique namespace for this training job
- `trainer.s3_base_path` - S3 base path for trainer checkpoints

## How to Enable MTC in Main VERL

If you prefer to use the main verl repository instead of the fork, you can enable MTC support by making the following code modifications:

### Prerequisites

Install the required MTC library in your training environment:

```bash
pip install amzn-sagemaker-checkpointing s3torchconnector tenacity boto3
```

Or use the provided `Dockerfile` in this directory which includes these dependencies.

### Code Modifications

#### 1. Modify `verl/utils/checkpoint/fsdp_checkpoint_manager.py`

**Add imports at the top of the file:**

```python
# MTC imports
from amzn_sagemaker_checkpointing.config.sagemaker_checkpoint_config import SageMakerCheckpointConfig
from amzn_sagemaker_checkpointing.checkpointing.filesystem.filesystem import (
    SageMakerTieredStorageWriter,
    SageMakerTieredStorageReader
)
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict
import torch.distributed as dist
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
```

**Add the `CheckpointState` class:**

```python
class CheckpointState(Stateful):
    """Wrapper class for model, optimizer, and scheduler state for DCP."""
    
    def __init__(self, model, optimizer=None, lr_scheduler=None, rng_state_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.rng_state_fn = rng_state_fn

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler else None
        extra_state = {
            "lr_scheduler": lr_scheduler_state_dict,
            "rng": self.rng_state_fn() if self.rng_state_fn else None,
        }
        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "extra": extra_state
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        if self.optimizer and state_dict.get("optimizer") is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.lr_scheduler and state_dict.get("extra", {}).get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(state_dict["extra"]["lr_scheduler"])
        if self.rng_state_fn and state_dict.get("extra", {}).get("rng") is not None:
            rng_state = state_dict["extra"]["rng"]
            self.rng_state_fn(rng_state)
```

**Replace the `load_checkpoint` method:**

```python
def load_checkpoint(self, local_path: str, hdfs_path: str = None, s3_base_path: str = None,
                    ckpt_namespace: str = None, del_local_after_load=False):
    """
    Load an FSDP checkpoint using MTC.
    
    Args:
        local_path: Local directory for checkpoint files
        hdfs_path: Unused (for API compatibility)
        s3_base_path: S3 path for persistent storage (e.g., "s3://bucket/checkpoints")
        ckpt_namespace: Unique namespace for this training job
        del_local_after_load: Remove local files after loading
    """
    if local_path is None:
        return

    # Setup SageMaker Tiered Storage Reader
    smcheckpointconfig = SageMakerCheckpointConfig(
        namespace=ckpt_namespace,
        world_size=torch.distributed.get_world_size(),
        s3_tier_base_path=s3_base_path,
        logger=logger,
        save_to_s3=True
    )
    checkpoint_reader = SageMakerTieredStorageReader(
        checkpoint_config=smcheckpointconfig
    )

    # Create state wrapper
    state = CheckpointState(
        model=self.model,
        optimizer=self.optimizer if self.should_load_optimizer else None,
        lr_scheduler=self.lr_scheduler if self.should_load_extra else None,
        rng_state_fn=self.load_rng_state if self.should_load_extra else None
    )

    # Find latest checkpoint
    checkpoint_id = None
    if os.path.exists(local_path):
        candidates = [d for d in os.listdir(local_path) if d.startswith("step_")]
        if candidates:
            checkpoint_id = sorted(candidates)[-1]

    # Perform distributed checkpoint load
    load_future = dcp.async_load(
        state_dict={"app": state},
        storage_reader=checkpoint_reader,
        checkpoint_id=checkpoint_id,
    )
    load_future.result()
    torch.distributed.barrier()
```

**Replace the `save_checkpoint` method:**

```python
def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0,
                    s3_base_path: str = None, ckpt_namespace: str = None, max_ckpt_to_keep=None):
    """
    Save an FSDP checkpoint using MTC.
    
    Args:
        local_path: Local directory for checkpoint files
        hdfs_path: Unused (for API compatibility)
        global_step: Current training step
        s3_base_path: S3 path for persistent storage
        ckpt_namespace: Unique namespace for this training job
        max_ckpt_to_keep: Number of recent checkpoints to retain
    """
    if local_path is None:
        return

    # Create SageMaker Checkpoint Config
    smcheckpointconfig = SageMakerCheckpointConfig(
        namespace=ckpt_namespace,
        world_size=torch.distributed.get_world_size(),
        s3_tier_base_path=s3_base_path,
        logger=logger,
        save_to_s3=False  # Async save to memory tier first
    )
    
    # Create tiered storage writer
    checkpoint_writer = SageMakerTieredStorageWriter(
        checkpoint_config=smcheckpointconfig,
        step=global_step
    )

    # Create state wrapper
    state = CheckpointState(
        model=self.model,
        optimizer=self.optimizer if self.should_save_optimizer else None,
        lr_scheduler=self.lr_scheduler if self.should_save_extra else None,
        rng_state_fn=self.get_rng_state if self.should_save_extra else None
    )

    # Await previous async save
    if hasattr(self, "checkpoint_future") and self.checkpoint_future is not None:
        exc = self.checkpoint_future.exception()
        if exc:
            print(f"Failure in saving previous checkpoint: {str(exc)}")
        else:
            result = self.checkpoint_future.result()

    # Perform async save
    checkpoint_id = f"step_{global_step}"
    self.checkpoint_future = dcp.async_save(
        state_dict={"app": state},
        storage_writer=checkpoint_writer,
        checkpoint_id=checkpoint_id,
    )
    
    torch.distributed.barrier()
```

#### 2. Modify `verl/utils/checkpoint/checkpoint_manager.py`

Update the base checkpoint manager to accept MTC parameters:

```python
def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0,
                    s3_base_path: str = None, ckpt_namespace: str = None, max_ckpt_to_keep=None):
    """Base save method with MTC parameter support."""
    raise NotImplementedError

def load_checkpoint(self, local_path: str, hdfs_path: str = None, s3_base_path: str = None,
                    ckpt_namespace: str = None, del_local_after_load=False):
    """Base load method with MTC parameter support."""
    raise NotImplementedError
```

#### 3. Modify `verl/workers/fsdp_workers.py`

Update the `save_checkpoint` method to pass MTC parameters:

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
    """Save checkpoint with MTC support."""
    self.checkpoint_manager.save_checkpoint(
        local_path=local_path,
        hdfs_path=hdfs_path,
        global_step=global_step,
        s3_base_path=self.config.actor.checkpoint.get('s3_base_path', None),
        ckpt_namespace=self.config.actor.checkpoint.get('ckpt_namespace', None),
        max_ckpt_to_keep=max_ckpt_to_keep
    )
```

#### 4. Modify `verl/trainer/ppo/ray_trainer.py`

Update critic checkpoint saving to include MTC parameters:

```python
# In the checkpoint saving section, update calls to:
self.critic_wg.save_checkpoint(
    critic_local_path,
    critic_remote_path,
    self.global_steps,
    s3_base_path=self.config.trainer.get('s3_base_path', None),
    ckpt_namespace=self.config.trainer.get('ckpt_namespace', None),
    max_ckpt_to_keep=max_critic_ckpt_to_keep
)
```

#### 5. Configuration Changes

Add MTC configuration to your training config:

```yaml
actor_rollout_ref:
  actor:
    checkpoint:
      s3_base_path: "s3://your-bucket/checkpoints"  # Required for MTC
      ckpt_namespace: "my-training-job"              # Required for MTC

trainer:
  s3_base_path: "s3://your-bucket/checkpoints"      # For critic checkpoints
  ckpt_namespace: "my-training-job"
```

Or via command line:

```bash
actor_rollout_ref.actor.checkpoint.s3_base_path=s3://your-bucket/checkpoints \
actor_rollout_ref.actor.checkpoint.ckpt_namespace=my-training-job \
trainer.s3_base_path=s3://your-bucket/checkpoints
```

### Key Differences from Original verl

| Feature | Original verl | MTC-enabled verl |
|---------|---------------|------------------|
| Checkpoint Format | PyTorch state_dict files | Distributed Checkpointing (DCP) format |
| Storage Backend | Local disk + HDFS | Tiered storage (Memory -> S3) |
| Save Operation | Synchronous | Asynchronous with futures |
| Loading | Direct file loading | Via `SageMakerTieredStorageReader` |



## Cleanup

If you need to stop a running job:
```bash
# List all jobs
ray job list --address http://localhost:8265

# Stop a specific job
ray job stop <job-id> --address http://localhost:8265

# Clean up completed/failed jobs
ray job delete <job-id> --address http://localhost:8265
```

To remove old checkpoints from S3:
```bash
aws s3 rm ${S3_CHECKPOINT_BASE} --recursive
```

```bash
# Delete the RayCluster
kubectl delete raycluster mtc-grpo-cluster
```
