# Train GPT3 NemoMegatron on Slurm with NeMo Framework Launcher <!-- omit from toc -->

> [!IMPORTANT]  
> NeMo Framework Launcher is compatible with NeMo version 1.0 only.

This project provides a guide to run [NemoMegatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/megatron.html) on AWS using a container from Nvidia GPU Cloud (NGC). The test cases in this case cover NemoMegatron for different model sizes: [126M](1.bmk-pretrain-gpt3-126m.sh), [5B](1.bmk-pretrain-gpt3-5b.sh), [40B](1.bmk-pretrain-gpt3-40b.sh) and [175B](1.bmk-pretrain-gpt3-175b.sh) parameters. The test cases can be executed on Slurm and use Nvidia Enroot and Nvidia Pyxis.

Table of contents:

- [1. Pre-requisites](#1-pre-requisites)
- [2. Build AWS-optimized Nemo-Launcher image](#2-build-aws-optimized-nemo-launcher-image)
- [3. Set-up the NemoMegatron environment](#3-set-up-the-nemomegatron-environment)
- [4. Prepare Input Data](#4-prepare-input-data)
- [5. Pre-training GPT3](#5-pre-training-gpt3)
- [6. Customizing Pre-Training](#6-customizing-pre-training)
- [7. Pre-Training llama2](#7-pre-training-llama2)
- [8. References](#8-references)
- [9. Authors / Reviewers](#9-authors--reviewers)

## 1. Pre-requisites

The following pre-requisites are needed to run this example:

- You are using p4de.24xlarge instances with A100 80GB or newer, with at least 80GB of memory per GPU.
- You have access to the base image [NeMo Framework Training](https://registry.ngc.nvidia.com/orgs/ea-bignlp/teams/ga-participants/containers/nemofw-training). To gain access to this image, go to [Get Access to NeMo Framework](https://developer.nvidia.com/nemo-framework) to enroll to organization/team `ea-bignlp/ga-participant`.
- Docker, [Enroot](https://github.com/NVIDIA/enroot) and [Pixys](https://github.com/NVIDIA/pyxis) installed on the cluster and available on all nodes. It is assumed you are using a Custom AMI ([example](../../ami_and_containers/1.amazon_machine_image))

You will need to setup the following environment variables before running the scripts. :

```bash
export NEMO_VERSION=23.11
export REPO=aws-nemo-megatron
export TAG=$NEMO_VERSION
export TARGET_PATH=/fsx/nemo-launcher-$NEMO_VERSION   # must be a shared filesystem
export TEST_CASE_PATH=/home/ec2-user/2.nemo-launcher  # where you copy the test case or set to your test case path
export ENROOT_IMAGE=/fsx/${REPO}_${TAG}.sqsh
cd $TEST_CASE_PATH
```

## 2. Build AWS-optimized Nemo-Launcher image

You will retrieve the container image from Nvidia, build an optimized container for EFA and, convert it into an Enroot file so we can run it on our cluster.

1. You have a registered account with Nvidia and can access NGC. Retrieve the NGC API key following [instructions from Nvidia](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key) and request access [here](https://developer.nvidia.com/nemo-framework/join) in order to be able to pull NeMo images.
2. Configure NGC as shown below using the command below, when requested use `$oauthtoken` for the login and the API key from NGC fro the password.

```bash
docker login nvcr.io
```

3. Copy the file `0.NemoMegatron-aws-optimized.Dockerfile` to the local directory and run the command below. Docker will retrieve the NemoMegatron container image from NGC then build an optimized container for AWS. This stage takes a few minutes and you can follow progress

```bash
docker build --progress plain -t ${REPO}:${TAG} -f 0.NemoMegatron-aws-optimized.Dockerfile .
```

4. Convert the Docker container image to an [Enroot](https://github.com/NVIDIA/enroot) squash file that will be stored in `/fsx`. This step takes a few minutes.

```bash
[[ -e $ENROOT_IMAGE ]] && rm $ENROOT_IMAGE ; /usr/bin/time enroot import -o $ENROOT_IMAGE dockerd://${REPO}:${TAG}
```

The Enroot squash file will be placed into the `/fsx` directory, backed by FSx Lustre to provide high read throughput by multiple compute nodes upon job starts.

## 3. Set-up the NemoMegatron environment

You will setup the target directory to host the configurations and requirements for NemoMegatron. It is assumed that your have an FSx for Lustre file system available to all nodes of your cluster via the mountpoint `/fsx`. We follow the same logic as in the [NemoMegatron Launcher documentation](https://github.com/NVIDIA/NeMo-Megatron-Launcher/tree/23.07#5111-slurm)

1. Create the target directory with the command below:

```bash
mkdir -p $TARGET_PATH
```

2. Retrieve files from the container and place them in the target directory. You execute the container on your head-node for this task using Enroot [start](https://github.com/NVIDIA/enroot/blob/master/doc/cmd/start.md) command.

```bash
cd $TARGET_PATH
enroot start --mount $TARGET_PATH:/workspace/mount_dir \
             --env NVIDIA_VISIBLE_DEVICES=void \
             $ENROOT_IMAGE \
             cp -a /opt/NeMo-Megatron-Launcher/launcher_scripts /opt/NeMo-Megatron-Launcher/auto_configurator /opt/nemo-data-curator /workspace/mount_dir/
```

The `NVIDIA_VISIBLE_DEVICES` variable is set to `void` to prevent the process to check for the Nvidia driver presence (since we don't need GPUs here).

3. Install the NemoMegatron requirements in a Python VirtualEnv by running the set of commands below.

```bash
cd $TARGET_PATH
sudo amazon-linux-extras install -y python3.8 # we need Python =>3.8
/usr/bin/python3.8 -m venv .venv
source .venv/bin/activate
pip3.8 install --upgrade pip setuptools
pip3.8 install -r <(curl -fsSL "https://raw.githubusercontent.com/NVIDIA/NeMo-Megatron-Launcher/${NEMO_VERSION%.*}/requirements.txt")
```

Next, you need to prepare the configuration files as follow:

1. Review and update the partition name in the .yaml config file `$TEST_CASE_PATH/conf.template/cluster/bcm.yaml`. Here is a summary of the values.

| Value              | Default                       | Definition                                                                                                                                                                  |
| ------------------ | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `partition`        | `null`                        | Slurm partition, same as a job queue                                                                                                                                        |
| `account`          | `null`                        | Account if using [accounting](https://slurm.schedmd.com/accounting.html)                                                                                                    |
| `exclusive`        | `True`                        | The job has [exclusive](https://stackoverflow.com/questions/66817279/what-does-the-keyword-exclusive-mean-in-slurm) use the instances it runs on (no other job can take it) |
| `gpus_per_task`    | `null`                        | Number of instances of GPUs per job                                                                                                                                         |
| `gpus_per_node`    | `8`                           | Number of GPUs to use per node. This is set to 8 GPUs as for th p4d.24xlarge                                                                                                |
| `mem`              | `0`                           | Requested memory (all)                                                                                                                                                      |
| `job_name_prefix`  | `"nemo-megatron-"`            | Prefix for your job names                                                                                                                                                   |
| `gres`             | `"gpu:8"`                     | Generic resource [scheduling](https://slurm.schedmd.com/gres.html)                                                                                                          |
| `srun_args`        | `"--no-container-mount-home"` | Arguments for the [srun](https://slurm.schedmd.com/srun.html) command (here for Pyxis)                                                                                      |
| `srun_args`        | `"-l"`                        | Arguments for the [srun](https://slurm.schedmd.com/srun.html) command to improve log verbosity. This flag prepends Slurm task number to lines of stdout/stderr.             |
| `srun_args`        | `"--open-mode=append"`        | Arguments for the [srun](https://slurm.schedmd.com/srun.html) command. When a Slurm job has multiple steps, each step appends (rather than truncate) to existing logs.      |
| `stderr_to_stdout` | `True`                        | Merge `stderr` and `stdout`                                                                                                                                                 |

2. Copy all the .yaml config files `{conf.template/ => launcher_scripts/conf/}` and substitute environment variables as follows:

```bash
cp -v ${TEST_CASE_PATH}/conf.template/cluster/* ${TARGET_PATH}/launcher_scripts/conf/cluster/
envsubst < ${TEST_CASE_PATH}/conf.template/config.yaml > ${TARGET_PATH}/launcher_scripts/conf/config.yaml
```

## 4. Prepare Input Data

The pre-training we're going to run uses the [GPT2](https://huggingface.co/gpt2) tokenizer which requires you to download the vocabularies files:

```bash
mkdir -p $TARGET_PATH/data/bpe
curl -L https://huggingface.co/gpt2/raw/main/vocab.json > $TARGET_PATH/data/bpe/vocab.json
curl -L https://huggingface.co/gpt2/raw/main/merges.txt > $TARGET_PATH/data/bpe/merges.txt
```

That's all needed to pre-train with a mock dataset generated on-the-fly.

## 5. Pre-training GPT3

This section assumes that you went through the previous sections and 1/ retrieved and built the AWS optimized NemoMegatron container, 2/ setup the NemoMegatron environment, and 3/ download the vocabularies. Here you start a pre-training on a small model of 126M parameters, this serves as a quick sanity check.

1. Source the NemoMegatron environment created earlier.

    ```bash
    source ${TARGET_PATH}/.venv/bin/activate
    ```

2. To pre-train a GPT3-126m on two instances with mock dataset, run the commands below to let :

    ```bash
    cd $TARGET_PATH
    $TEST_CASE_PATH/1.bmk-pretrain-gpt3-126m.sh
    ```

3. Check the file `$TARGET_PATH/launcher_scripts/main.py`. The `launcher_scripts/main.py` interacts with Slurm on our behalf to generate an `.sbatch` file and submits it to Slurm. Nemo-launcher logs all the invocation commands, output, and error to `$TARGET_PATH/results/<MODEL_SIZE>/` described below.

    ```bash
    $TARGET_PATH/results/gpt3_126m
    ├── gpt3_126m_hydra.yaml                        # The fully interpolated pre-training configuration
    ├── launcher_cmd.log                            # The full invocation command of launcher_scripts/main.py
    ├── launcher.log                                # Job id produced by the sbatch command
    ├── log-nemo-megatron-gpt3_126m_<JOB_ID>.out    # Stdout of the pre-training Slurm job
    ├── nemo-megatron-gpt3_126m_submission.sh       # .sbatch file generated and submitted by nemo-launcher
    └── results
        ├── cmd-args.log                            # The full invocation command of the pre-training script
        ├── events.out.tfevents.*                   # Tensorboard logs
        ├── git-info.log                            # The commit hash of the NeMO repo provided in the container.
        ├── hparams.yaml                            # Pre-training hyperparameters
        ├── lightning_logs.txt                      # Additional logs from PyTorch-Lightning
        ├── nemo_error_log.txt                      # Stderr of pre-training step
        └── nemo_log_globalrank-*.txt               # Log of each rank
    ```

    Please note that except for `log-nemo-megatron-gpt3_126m_<JOB_ID>.out`, the other files will be overridden when you launch another pre-training of that same model size. To completely separate the output among jobs, run the script in benchmark mode: `BMK_MODE=1 $TEST_CASE_PATH/bmk-pretrain-gpt3-126m.sh` which produces output dir `$TARGET_PATH/results-<YYYYMMDD>-<HHMMSS>utc-<RANDOM_STR>/gpt3_126m/`.

4. You can use Slurm command `squeue` to monitor the job status in the queue. The ample output below shows a `nemo-megatron` job with job id `1234` is in running state (`ST` = `R`). A queued job will have state `ST` = `PD` (pending). Please refer to the complete of job states in this [Slurm documentation](https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES).

    ```text
    JOBID   PARTITION        NAME      USER  ST       TIME  NODES NODELIST(REASON)
     1234   my-cluste   nemo-mega  ec2-user   R   00:19:40      1 p4de-dy-p4de-24xlarge-[1-2]
    ```

5. Once a job finishes, check the `log-nemo-megatron-<MODEL_NAME>_<MODEL_SIZE>_<JOB_ID>.err`, and see it should contains ``Trainer.fit` stopped: `max_steps=40` reached`` (disregard the warnings).

    ```console
    $ tail -5 $TARGET_PATH/results/gpt3_126m/log-nemo-megatron-gpt3_126m_72.err

    [NeMo W 2023-09-11 22:31:45 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:232: UserWarning: You called `self.log('consumed_samples', ...)` in your `training_step` but the value needs to be floating point. Converting it to torch.float32.
          warning_cache.warn(

    `Trainer.fit` stopped: `max_steps=40` reached.
    ```

6. Review the output file (`log-nemo-megatron-gpt3_126m_<JOB_ID>.out`) which contains the `stdout` output of the job. The end of the file should be similar to the snippet below

    ```console
    [NeMo I 2023-09-11 22:31:28 lr_scheduler:910] Scheduler "<nemo.core.optim.lr_scheduler.CosineAnnealing object at 0x7f8ffd427490>"
        will be used during training (effective maximum steps = 40) -
        Parameters :
        (warmup_steps: 636
        constant_steps: 100000
        min_lr: 6.0e-05
        max_steps: 40
        )
    Epoch 0: 100%|██████████| 40/40 [00:31<00:00,  1.27it/s, loss=10.9, v_num=, reduced_train_loss=10.90, global_step=39.00, consumed_samples=9984.0]
    ```

Congratulations! You've successfully run this test case to completion.

> **Note**: Should you run into an OOM error, you can adjust the minimum batch size by setting the MBS in `bmk` launch scripts. You can tune the NemoMegatron and PyTorch parameters in such way as well.

## 6. Customizing Pre-Training

To pre-train for a different model size on different instance count, open `$TEST_CASE_PATH/1.bmk-pretrain-gpt3-126m.sh` and edit section `000` to choose the right hyperparameters. Be aware that pre-training LLM requires understanding on the hyperparameters such as parallelism and batches. Please refer to the NeMO project ([website](https://developer.nvidia.com/nemo), [GitHub](https://github.com/NVIDIA/NeMo), [NeMo-Megatron-Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher)) and the Megatron papers ([Shoeybi20](https://arxiv.org/abs/1909.08053), [Narayanan21](https://arxiv.org/abs/2104.04473)).

At the very least, you'd want to review and customize one or more YAML files under `$TARGET_PATH/launcher_scripts/conf/`. Nemo-launcher organizes its config files in an opinionated hierarchy. Below is an example of relevant YAML files when launching `$TARGET_PATH/launcher_scripts/main.py` for `training` stage for `gpt3/126m` (see `$TEST_CASE_PATH/1.bmk-pretrain-gpt3-126m.sh`).

```bash
$TARGET_PATH/launcher_scripts/conf
├── config.yaml        # Config for generating job scripts (.sbatch, .yaml, etc.)
├── cluster
│   └── bcm.yaml       # Config for Slurm jobs
└── training           # Config for stage "training"
    └── gpt3           # Config for model "gpt3"
        └── 126m.yaml  # Config for model size "126m"
```

You can edit directly the `gpt3/<MODEL_SIZE>.yaml` to customize the number of instances, tensor parallelism, pipeline parallelism, batch sizes (micro and global), experiment tracking, etc. on this file. Alternatively, you can override the settings through the CLI options of `$TARGET_PATH/launcher_scripts/main.py` (refer to `1.bmk-pretrain-gpt3-126m.sh`). For example, this CLI arg `training.trainer.num_nodes=$NUM_NODES` is equivalent to editing file `$TARGET_PATH/launcher_scripts/training_scripts/conf/training/<MODEL_NAME>/<MODEL_SIZE>.yaml` to set key `trainer -> num_nodes` to `$NUM_NODES`.

```text
    +-- file `training/<MODEL_NAME>/<MODEL_SIZE>.yaml` under `$TARGET_PATH/launcher_scripts/conf`
    |
/---+--\
training.trainer.num_nodes=$NUM_NODES
         \_______________/
                |
                └── key 'trainer -> num_nodes' in the `.yaml` file.
```

## 7. Pre-Training llama2

This section assumes that you went through the previous sections and 1/ retrieved and built the AWS optimized NemoMegatron container, 2/ setup the NemoMegatron environment, and 3/ download the vocabularies. Actions will be almost the same as for 5/ Pre-training GPT3, let do it.

1. Download llama2 tokenizer

```
mkdir -p $TARGET_PATH/data/llama2
curl -L https://github.com/microsoft/Llama-2-Onnx/raw/main/tokenizer.model > $TARGET_PATH/data/llama2/tokenizer.model

```

2. Source the NemoMegatron environment created earlier.

    ```bash
    source ${TARGET_PATH}/.venv/bin/activate
    ```

3. To pre-train a llama2-7b on two instances with mock dataset, run the commands below to let :

    ```bash
    cd $TARGET_PATH
    $TEST_CASE_PATH/5.bmk-pretrain-llama-7b.sh
    ```

4. Next stests are absolutely the same as for 5/ Pre-training GPT3, the only difference is that result directory is `$TARGET_PATH/results/llama2_7b`

## 8. References

- Nvidia NemoMegatron Documentation: <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/megatron.html>
- Train Large Scale NLP with Nemo Megatron from Nvidia: <https://docs.nvidia.com/launchpad/ai/base-command-nemo/latest/index.html>

## 9. Authors / Reviewers

- [A] Verdi March - marcverd@
- [R] Pierre-Yves Aquilanti - pierreya@
