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
from nemo.lightning.run import plugins
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.llm.recipes.callbacks.common import straggler_det_callback
from nemo.lightning.pytorch.callbacks import PreemptionCallback
from nemo.lightning.run import plugins
from nemo.utils import logging


# python finetune.py --max_steps 200 --nodes 1 --gpus L40S --gpu-devices 8 --container_image nvcr.io/nvidia/nemo:26.02 --env_vars_file env_vars.json --pvc_name fsx-claim --pvc_mount_path /mnt/nemo


def get_parser():
   parser = argparse.ArgumentParser(description="NeMo 2.0 on SageMaker Hyperpod EKS")
   parser.add_argument("--nodes", type=int, help="Number of nodes to run on", default=1)
   parser.add_argument("--gpus", type=str, help="GPU type to use", default="L40S")
   parser.add_argument("--gpu-devices", type=int, help="Number of GPUs per node", default=8)
   parser.add_argument("--efa-devices", type=int, help="Number of EFA devices per node", default=None)
   parser.add_argument("--max_steps", type=int, help="Maximum number of steps", default=200)
   parser.add_argument("--container_image", type=str, help="Container image to use", default="nvcr.io/nvidia/nemo:26.02")
   parser.add_argument("--env_vars_file", type=str, help="Path to the JSON file with environment variables", default="env_vars.json")
   parser.add_argument("--pvc_name", type=str, help="Name of the Persistent Volume Claim to use", default="fsx-claim")
   parser.add_argument("--pvc_mount_path", type=str, help="Path where the PVC should be mounted in the container", default="/mnt/nemo")
   parser.add_argument("--hf_token", type=str, help="Hugging Face token", default=None)
   parser.add_argument("--disable_lora", action="store_true", help="Disable LoRA finetuning (LoRA is enabled by default)")
   return parser

def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model=llm.gemma2_2b.model(),
        source="hf://google/gemma-2-2b",
        overwrite=False
    )

def configure_finetune_recipe(
    exp_name=None,
    work_dir=None,
    peft_scheme=None,
    lora_enabled=None,
    max_steps=None,
    nodes=None,
    gpu_devices=None,
    ):
   finetune_recipe = llm.gemma2_2b.finetune_recipe(
       num_nodes=nodes,
       name=exp_name,
       dir=work_dir,
       peft_scheme=peft_scheme
    )
   finetune_recipe.trainer.devices = gpu_devices
   finetune_recipe.trainer.num_sanity_val_steps = 0
   finetune_recipe.trainer.max_steps = max_steps
   finetune_recipe.trainer.strategy.context_parallel_size = 1
   finetune_recipe.trainer.val_check_interval = 10

   if lora_enabled:
       finetune_recipe.trainer.strategy.ddp = "megatron"
   
   return finetune_recipe


def skypilot_executor(
   nodes: int,
   pvc_mount: str,
   gpu_devices: int,
   gpus: str = "L40S",
   efa_devices: Optional[int] = None,
   custom_mounts: Optional[dict[str, str]] = None,
   container_image: str = "nvcr.io/nvidia/nemo:26.02",
   env_vars_file: str = "env_vars.json",
   pvc_name: str = "nemo-runs",
   lora_enabled: bool = False,
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
   executor.env_vars["NEMO_HOME"] = f"{pvc_mount}/nemo"
   executor.env_vars["NEMO_MODELS_CACHE"] = f"{pvc_mount}/nemo/cache"
   executor.env_vars["HF_HOME"] = f"{pvc_mount}/huggingface"
   executor.env_vars["HF_HUB_CACHE"] = f"{pvc_mount}/huggingface/hub"
   if args.hf_token:
       executor.env_vars["HF_TOKEN"] = args.hf_token
   else:
       logging.info("No Hugging Face token provided, gated repositories may be inaccessible.")

   return executor


if __name__ == "__main__":
   args = get_parser().parse_args()
       
   pvc_mount = args.pvc_mount_path
   work_dir  = os.path.join(pvc_mount, "experiments")

   # LoRA is enabled by default, disabled if --disable_lora flag is present
   lora_enabled = not args.disable_lora
   
   if lora_enabled:
       peft_scheme = "lora"
       exp_name = f"aws-nemo2-lora-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
   else:
       peft_scheme = None
       exp_name = f"aws-nemo2-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

   import_ckpt = configure_checkpoint_conversion()
   finetune_recipe = configure_finetune_recipe(
       exp_name=exp_name,
       work_dir=work_dir,
       peft_scheme=peft_scheme,
       lora_enabled=lora_enabled,
       max_steps=args.max_steps,
       nodes=args.nodes,
       gpu_devices=args.gpu_devices
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
       pvc_mount=pvc_mount,
       lora_enabled=lora_enabled
   )
   # Set launcher based on number of nodes
   if args.nodes > 1:
       executor.launcher = "torchrun"
   
   # Set up the executor for the checkpoint conversion
   import_executor = executor.clone()

   with run.Experiment(exp_name, log_level="INFO") as exp:
       exp.add(import_ckpt, executor=import_executor, name="checkpoint_conversion")
       exp.add(
          finetune_recipe, 
          executor=executor, 
          tail_logs=True, 
          name="finetuning"
       )
       # Run the experiment
       exp.run(sequential=True, detach=True)