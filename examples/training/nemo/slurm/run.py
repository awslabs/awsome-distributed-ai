import nemo_run as run
import json
import argparse
import math
import os
import signal
from functools import partial
from typing import Any, Optional
from nemo.collections import llm
from nemo.lightning.run import plugins
from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.llm.recipes.callbacks.common import straggler_det_callback
from nemo.lightning.pytorch.callbacks import PreemptionCallback
from nemo.lightning.run import plugins


# python run.py --nodes 2 --max_steps 1000


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
   parser = argparse.ArgumentParser(description="NeMo 2.0 on SageMaker Hyperpod")
   parser.add_argument("--partition", type=str, help="Slurm partition to run on", default="dev")
   parser.add_argument("--nodes", type=int, help="Number of nodes to run on", default=1)
   parser.add_argument("--max_steps", type=int, help="Maximum number of steps", default=200)
   parser.add_argument("--account", type=str, help="Slurm account to use", default="ubuntu")
   parser.add_argument("--container_image", type=str, help="Container image to use", default="/fsx/ubuntu/aws-nemo-26-02.sqsh")
   parser.add_argument("--time", type=str, help="Time to run the job", default="01:00:00")
   parser.add_argument("--env_vars_file", type=str, help="Path to the JSON file with environment variables", default="env_vars.json")
   parser.add_argument("--ntasks_per_node", type=int, help="Number of tasks per node", default=8)

   return parser


def slurm_executor(
   account: str,
   partition: str,
   nodes: int,
   user: str = "local",
   host: str = "local",
   remote_job_dir: str = "/fsx/ubuntu/aws-nemo",
   time: str = "01:00:00",
   custom_mounts: Optional[list[str]] = None,
   container_image: str = "/fsx/ubuntu/aws-nemo-26-02.sqsh",
   env_vars_file: str = "env_vars.json",
   ntasks_per_node: int = 8,
   retries: int = 0,
) -> run.SlurmExecutor:
   if not (user and host and remote_job_dir and account and partition and nodes):
       raise RuntimeError(
           "Please set user, host, remote_job_dir, account, partition, nodes args for using this function."
       )


   # mounts = ["/a/b/c:/b/c"]
   mounts = []
   # Custom mounts are defined here.
   if custom_mounts:
       mounts.extend(custom_mounts)
   # Env vars for jobs are configured here
   with open(env_vars_file, 'r') as f:
       env_vars = json.load(f)   

   # This will package the train.py script in the current working directory to the remote cluster.
   # If you are inside a git repo, you can also use https://github.com/NVIDIA/NeMo-Run/blob/main/src/nemo_run/core/packaging/git.py.
   # If the script already exists on your container and you call it with the absolute path, you can also just use `run.Packager()`.
   packager = run.Packager()


   local_tunnel = run.LocalTunnel(job_dir="")
   srun_args = None
   if os.path.isdir("/opt/sagemaker_cluster"):
       print("Detected Hyperpod cluster.. enabling --auto-resume=1")
       srun_args = ["--auto-resume=1"]
  
   # This defines the slurm executor.
   # We connect to the executor via the tunnel defined by user, host and remote_job_dir.
   # https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemorun/guides/execution.html#slurmexecutor
   # https://github.com/NVIDIA/NeMo-Run/blob/7bca60a6d2af3eea1aba331d3a80a6d7351b6e84/nemo_run/core/execution/slurm.py#L99
   executor = run.SlurmExecutor(
       account=account,
       partition=partition,
       tunnel=local_tunnel,
       nodes=nodes,
       ntasks_per_node=ntasks_per_node,
       mem="0",
       exclusive=True,
       packager=packager,
       srun_args=srun_args,
   )


   executor.container_image = container_image
   executor.container_mounts = mounts
   executor.env_vars = env_vars
   executor.retries = retries
   executor.time = time


   return executor


if __name__ == "__main__":
   args = get_parser().parse_args()
  
   import random, string
   stri = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
   exp_name = f"aws-nemo2"+stri
  
   pretrain_recipe = partial(llm.llama31_8b.pretrain_recipe, num_nodes=args.nodes)(name=exp_name, dir="")
   pretrain_recipe.trainer.callbacks.append(straggler_det_callback(straggler_report_time_interval=4))
   pretrain_recipe.trainer.strategy.ckpt_async_save = True
   pretrain_recipe.trainer.num_sanity_val_steps = 0
   pretrain_recipe.model = run.Config(llm.LlamaModel, small_llama_cfg())
   pretrain_recipe.data.tokenizer = run.Config(get_nmt_tokenizer, library="megatron", model_name= "GPT2BPETokenizer", vocab_file="/root/.cache/torch/megatron/megatron-gpt-345m_vocab", merges_file="/root/.cache/torch/megatron/megatron-gpt-345m_merges")
   pretrain_recipe.broadcast(max_steps=args.max_steps)
   pretrain_recipe.trainer.limit_val_batches = 2
   pretrain_recipe.trainer.log_every_n_steps = 1
   pretrain_recipe.trainer.val_check_interval = 10


   # Run it locally
   executor = slurm_executor(
       partition=args.partition,
       account=args.account,
       nodes=args.nodes,
       container_image=args.container_image,
       time=args.time,
       custom_mounts=[
           "~/megatron:/root/.cache/torch/megatron"
       ],
       env_vars_file=args.env_vars_file,
       ntasks_per_node=args.ntasks_per_node
   )

   executor.launcher = "ft"
   run_plugins: list[run.Plugin] = [
      plugins.PreemptionPlugin(callbacks=[run.Config(PreemptionCallback, sig=signal.SIGINT)]),
      plugins.FaultTolerancePlugin()
   ]
   

   with run.Experiment(exp_name, log_level="INFO") as exp:
       exp.add(
          pretrain_recipe, 
          executor=executor, 
          tail_logs=True, 
          name="training",
          plugins=run_plugins
       )
       # Run the experiment
       exp.run(detach=True)

