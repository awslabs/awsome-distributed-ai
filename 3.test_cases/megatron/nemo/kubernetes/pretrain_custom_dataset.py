import signal
import nemo_run as run
import json
import argparse
import math
import os
from datetime import datetime
from functools import partial
from typing import Any, Optional
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging
from datasets import load_dataset

# python pretrain.py --max_steps 200 --nodes 1 --gpus L40S --gpu-devices 8 --container_image nvcr.io/nvidia/nemo:26.02 --env_vars_file env_vars.json --pvc_name fsx-claim --pvc_mount_path /mnt/nemo


def small_llama_cfg() -> llm.GPTConfig:
   """Small 180m model"""
   return run.Config(
       llm.Llama3Config8B,
       rotary_base=500_000,
       seq_length=1024,
       num_layers=12,
       hidden_size=768,
       ffn_hidden_size=2688,
       num_attention_heads=16,
       init_method_std=0.023,
   )


def get_parser():
   parser = argparse.ArgumentParser(description="LLM Pretraining with NeMo 2.0 on Kubernetes")
   parser.add_argument("--nodes", type=int, help="Number of nodes to run on", default=1)
   parser.add_argument("--gpus", type=str, help="GPU type to use", default="L40S")
   parser.add_argument("--gpu-devices", type=int, help="Number of GPUs per node", default=8)
   parser.add_argument("--efa-devices", type=int, help="Number of EFA devices per node", default=None)
   parser.add_argument("--max_steps", type=int, help="Maximum number of steps", default=200)
   parser.add_argument("--container_image", type=str, help="Container image to use", default="nvcr.io/nvidia/nemo:26.02")
   parser.add_argument("--env_vars_file", type=str, help="Path to the JSON file with environment variables", default="env_vars.json")
   parser.add_argument("--pvc_name", type=str, help="Name of the Persistent Volume Claim to use", default="fsx-claim")
   parser.add_argument("--pvc_mount_path", type=str, help="Path where the PVC should be mounted in the container", default="/mnt/nemo")
   parser.add_argument("--seq_length", type=int, help="The sequence length of the dataset", default=1024)
   parser.add_argument("--micro_batch_size", type=int, help="The micro batch size of the dataset", default=2)
   parser.add_argument("--global_batch_size", type=int, help="The global batch size of the dataset (if not specified, will be calculated as devices * micro_batch_size)", default=None)

   return parser


def skypilot_executor(
   nodes: int,
   pvc_mount: str,
   gpu_devices: int,
   gpus: str = "L40S",
   efa_devices: Optional[int] = None,
   custom_mounts: Optional[dict[str, str]] = None,
   container_image: str = "nvcr.io/nvidia/nemo:26.02",
   env_vars_file: str = "env_vars.json",
   pvc_name: str = "nemo-runs"
) -> run.SkypilotExecutor:

   mounts = {}
   # Custom mounts are defined here.
   if custom_mounts:
       for k, v in custom_mounts.items():
           mounts[k] = v
   # Env vars for jobs are configured here
   with open(env_vars_file, 'r') as f:
       env_vars = json.load(f)

   packager = run.GitArchivePackager()

   shared_pod_config = {
        "kubernetes": {
            "pod_config": {
                "spec": {
                    "containers": [{ 
                        "volumeMounts": [
                            {"name": "nemo-runs", "mountPath": pvc_mount},
                        ]
                    }],
                    "volumes": [{
                        "name": "nemo-runs",
                        "persistentVolumeClaim": {"claimName": pvc_name}
                    }]
                }
            }
        }
    }

   if efa_devices is not None:
        shared_pod_config["kubernetes"]["pod_config"]["spec"]["containers"][0]["resources"] = {
            "requests": {
                "vpc.amazonaws.com/efa": efa_devices
            },
            "limits": {
                "vpc.amazonaws.com/efa": efa_devices
            }
        }
  
   # This defines the skypilot executor.
   executor = run.SkypilotExecutor(
       cloud="kubernetes",
       gpus=gpus,
       gpus_per_node=gpu_devices,
       num_nodes=nodes,
       packager=packager,
       cluster_config_overrides=shared_pod_config
   )

   executor.container_image = container_image
   executor.file_mounts = mounts
   executor.env_vars = env_vars
   executor.env_vars["NEMORUN_HOME"] = pvc_mount

   return executor


if __name__ == "__main__":
   args = get_parser().parse_args()

    # Calculate global batch size ensuring proper divisibility
   if args.global_batch_size is None:
       global_batch_size = args.nodes * args.gpu_devices * args.micro_batch_size
   else:
       global_batch_size = args.global_batch_size
       
   # Validate batch size alignment
   data_parallel_size = args.nodes * args.gpu_devices  # Total GPUs across all nodes
   if global_batch_size % (args.micro_batch_size * data_parallel_size) != 0:
       raise ValueError(
           f"Global batch size ({global_batch_size}) must be divisible by "
           f"micro_batch_size ({args.micro_batch_size}) * data_parallel_size ({data_parallel_size}) = "
           f"{args.micro_batch_size * data_parallel_size}. "
           f"Current configuration: {args.nodes} nodes × {args.gpu_devices} devices/node = {data_parallel_size} total GPUs"
       )
   
   exp_name = f"aws-nemo2-pretrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
   pvc_mount = args.pvc_mount_path
   work_dir = os.path.join(pvc_mount, "experiments")
  
   # Using custom dataset with the llama3 8B pretraining recipe
   pretrain_recipe = partial(llm.llama3_8b.pretrain_recipe, num_nodes=args.nodes)(name=exp_name, dir=work_dir)
   pretrain_recipe.trainer.devices = args.gpu_devices
   pretrain_recipe.trainer.num_sanity_val_steps = 0
   pretrain_recipe.model = run.Config(llm.LlamaModel, small_llama_cfg())
   pretrain_recipe.broadcast(max_steps=args.max_steps)
   pretrain_recipe.trainer.limit_val_batches = 2
   pretrain_recipe.trainer.log_every_n_steps = 1
   pretrain_recipe.trainer.val_check_interval = 10
   pretrain_recipe.data = run.Config(
        llm.PreTrainingDataModule,
        paths=[os.path.join(pvc_mount, "data", "processed_data", "train_dataset_text_document")],
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=global_batch_size,
        split="95,3,2",
        tokenizer=run.Config(
            get_nmt_tokenizer,
            library="megatron",
            model_name="GPT2BPETokenizer", 
            vocab_file=os.path.join(pvc_mount, "data", "tokenizer", "gpt2-vocab.json"), 
            merges_file=os.path.join(pvc_mount, "data", "tokenizer", "gpt2-merges.txt")
        )
    )

   executor = skypilot_executor(
       nodes=args.nodes,
       gpus=args.gpus,
       gpu_devices=args.gpu_devices,
       efa_devices=args.efa_devices,
       container_image=args.container_image,
       custom_mounts={
           "/root/nemo": "."
       },
       env_vars_file=args.env_vars_file,
       pvc_name=args.pvc_name,
       pvc_mount=pvc_mount
   )
    
      # Set launcher based on number of nodes
   if args.nodes > 1:
       executor.launcher = "torchrun"

   with run.Experiment(exp_name, log_level="INFO") as exp:
       exp.add(
          pretrain_recipe, 
          executor=executor, 
          tail_logs=True, 
          name="pretraining",
       )
       # Run the experiment
       exp.run(detach=True)