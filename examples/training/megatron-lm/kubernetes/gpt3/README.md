# GPT Model Training on EKS with MegatronLM

This directory contains Kubernetes-specific instructions and templates for setting up and running GPT model training using MegatronLM on an EKS cluster.

## 1. Preparation

Before proceeding with GPT3 training setup, please follow the steps described in [../README.md](../README.md) to prepare your environment
The following example assumes that you have a PVC named `fsx-claim` and the `REPO_URI` environment variable is exported.

### 1.1 Determine Compute Resources

Before running the training, you need to determine the compute resources available on your EKS cluster nodes. This will help you set the correct resource limits for GPUs and EFA (Elastic Fabric Adapter) network interfaces.

Export the following environment variables based on your instance type:

```bash
# Example for p5.48xlarge
export INSTANCE_TYPE=p5.48xlarge
export GPU_PER_NODE=8
export EFA_PER_NODE=32
export NUM_NODES=2
```

You can refer to the following table to find the correct values for your instance type:

| Instance Type | GPUs | EFA Interfaces 
|---------------|------|----------------
| p5.48xlarge   | 8    | 32             
| p5e.48xlarge  | 8    | 32             
| p5en.48xlarge | 8    | 16             
| p6-b200.48xlarge | 8 | 8             



### 2. Data Preprocessing

1. Run the following snippet to crete a job container that mounts the fsx volume and downloads the input datasets and vocabulary on it:

    #### Step 1: Create and Apply the Data Download Job

    Generate the `getdata-job.yaml` manifest from the template and apply it:

    ```bash
    envsubst < manifests/getdata-job.yaml-template > manifests/getdata-job.yaml
    kubectl apply -f manifests/getdata-job.yaml
    ```

    #### Step 2: Verify Job Creation

    List jobs to confirm creation:

    ```bash
    kubectl get jobs
    ```

    You should see an entry for `getdata-job` with information about its status, completions, and age. To get more details about the pods created by the job, run:

    ```bash
    kubectl get pods -l job-name=getdata-job
    ```

    This will show the pod(s) managed by the job. If you want to describe the job and see events or issues, use:

    ```bash
    kubectl describe job getdata-job
    ```

    #### Step 3: Monitor Job Progress

    Stream the logs to monitor download progress:

    ```bash
    kubectl logs -f job/getdata-job
    ```

    **Note:** You should be able to see output similar to the following once the downloads have completed successfully:

    ```text
    ...
    Saving to: 'gpt2-merges.txt'

         0K .......... .......... .......... .......... .......... 11% 19.2M 0s
        50K .......... .......... .......... .......... .......... 22% 55.9M 0s
       100K .......... .......... .......... .......... .......... 33% 57.3M 0s
       150K .......... .......... .......... .......... .......... 44% 66.1M 0s
       200K .......... .......... .......... .......... .......... 56%  106M 0s
       250K .......... .......... .......... .......... .......... 67%  132M 0s
       300K .......... .......... .......... .......... .......... 78%  139M 0s
       350K .......... .......... .......... .......... .......... 89%  133M 0s
       400K .......... .......... .......... .......... .....     100%  122M=0.007s

    2025-06-20 08:59:58 (62.9 MB/s) - 'gpt2-merges.txt' saved [456318/456318]

    total 940M
    drwxr-xr-x 2 root root   33K Jun 20 09:00 .
    drwxr-xr-x 5 root root   33K Jun 20 08:59 ..
    -rw-r--r-- 1 root root  446K Feb 18  2019 gpt2-merges.txt
    -rw-r--r-- 1 root root 1018K Feb 18  2019 gpt2-vocab.json
    -rw-r--r-- 1 root root  1.1G Jul 24  2021 oscar-1GB.jsonl
    Download completed.
    ```

    #### Step 5: Cleanup

    Once the job status is `Completed`, delete the job and its pod:

    ```bash
    kubectl delete -f manifests/getdata-job.yaml
    ```


2. Preprocess the data

    Launch the preprocessing job to convert the downloaded data for training.

    ```bash
    cat manifests/prepdata-job.yaml-template | envsubst > manifests/prepdata-job.yaml
    kubectl apply -f ./manifests/prepdata-job.yaml
    ```

    Check pods for `prepdata-job`:

    ```bash
    kubectl get pods -l job-name=prepdata-job
    ```

    Monitor the job's progress by streaming its logs:

    ```bash
    kubectl logs -f job/prepdata-job
    ```

    The expected log output from the above command should look similar to the following when preprocessing completes successfully:

    ```text
    ...
    -rw-r--r--  1 root root 3.4K Jun 14 02:55 pretrain_vision_classify.py
    -rw-r--r--  1 root root 3.5K Jun 14 02:55 pretrain_vision_dino.py
    -rw-r--r--  1 root root 4.8K Jun 14 02:55 pretrain_vision_inpaint.py
    -rw-r--r--  1 root root 8.2K Jun 14 02:55 pretrain_vlm.py
    -rw-r--r--  1 root root  824 Jun 14 02:55 pyproject.toml
    -rw-r--r--  1 root root 4.0K Jun 14 02:55 setup.py
    drwxr-xr-x  8 root root  200 Jun 14 02:55 tasks
    drwxr-xr-x  4 root root   67 Jun 14 02:55 tests
    drwxr-xr-x  6 root root 4.0K Jun 14 02:55 tools
    Data preprocessing completed.
    ```

    After the job status is `Completed`, clean up the job and its pod:

    ```bash
    kubectl delete -f prepdata-job.yaml
    ```

    Voilà! The preprocessing job has finished. You are now ready to proceed to the training step.

### 3. Distributed training

Now that the data is preprocessed, we will pretrain a GPT3 model MegatronLM.  Launch a PyTorchJob with the environment variables:

```bash
export TENSOR_PARALLEL=8
export PIPELINE_PARALLEL=1
export NUM_LAYERS=36
export HIDDEN_SIZE=4096
export NUM_ATTENTION_HEADS=32
export SEQ_LENGTH=2048
export MAX_POSITION_EMBEDDINGS=2048
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=288
cat manifests/pytorchjob.yaml-template | envsubst > manifests/pytorchjob.yaml
kubectl apply -f ./manifests/pytorchjob.yaml
```

The training starts running:

```bash
kubectl get pods
```

You should see one etcd and one worker pod.

```bash
NAME                    READY   STATUS      RESTARTS   AGE
etcd-7787559c74-wpcb9   1/1     Running     0          3m10s
megatron-worker-0       1/1     Running     0          3m10s
```

Log lines describing the iterations show that the training is working properly.

```bash
kubectl logs -f megatron-worker-0
```

An abbreviated sample log is shown below:

   An abbreviated sample log is shown below:
    
   ```text
   ...
   using torch.float16 for parameters ...
   ------------------------ arguments ------------------------
   accumulate_allreduce_grads_in_fp32 .............. False
   adam_beta1 ...................................... 0.9
   adam_beta2 ...................................... 0.95
   ...
   -------------------- end of arguments ---------------------
   setting number of micro-batches to constant 288
   > building GPT2BPETokenizer tokenizer ...
   > padded vocab (size: 50257) with 943 dummy tokens (new size: 51200)
   > initializing torch distributed ...
   > initialized tensor model parallel with size 8
   > initialized pipeline model parallel with size 1
   > setting random seeds to 1234 ...
   > compiling dataset index builder ...
   make: Entering directory '/workspace/Megatron-LM/megatron/core/datasets'
   ...
   time to initialize megatron (seconds): 15.424
   [after megatron is initialized] datetime: 2024-07-16 22:14:01
   building GPT model ...
   > number of parameters on (tensor, pipeline) model parallel rank (4, 0): 941594624
   ...
   > building train, validation, and test datasets ...
   > datasets target sizes (minimum size):
       train:      146484375
       validation: 5863680
       test:       11520
   ...
   iteration        1/  508626 | consumed samples:          288 | elapsed time per iteration (ms): 255940.5 | learning rate: 0.000E+00 | global batch size:   288 | loss scale: 4294967296.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
   iteration        2/  508626 | consumed samples:          576 | elapsed time per iteration (ms): 243438.3 | learning rate: 0.000E+00 | global batch size:   288 | loss scale: 2147483648.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
   iteration        3/  508626 | consumed samples:          864 | elapsed time per iteration (ms): 243344.4 | learning rate: 0.000E+00 | global batch size:   288 | loss scale: 1073741824.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
   ...
   ```
    
   You can stop the training job by executing:
    
   ```bash
   kubectl delete -f ./pytorchjob.yaml
   ```
    
## 4. What's next?

The example is based on the GPT3 example from MegatronLM's [repository](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/pretrain_gpt.sh). You can modify `NUM_ATTENTION_HEADS`, `NUM_LAYERS`, and `HIDDEN_SIZE`  based on the Table 1 (Page 8) of the document [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) to change the model size. You can also run the following commands to launch training for different model sizes before submitting a job as follows: `NUM_LAYERS=64 HIDDEN_SIZE=8192 NUM_ATTENTION_HEADS=48 sbatch  3.distributed-training.sbatch`

| Model size | Parameters                                                |
|------------|-----------------------------------------------------------|
| 1.7B       | `NUM_ATTENTION_HEADS=24 HIDDEN_SIZE=2304 NUM_LAYERS=24`   |
| 3.6B       | `NUM_ATTENTION_HEADS=32 HIDDEN_SIZE=3072 NUM_LAYERS=30`   |
| 7.5B       | `NUM_ATTENTION_HEADS=32 HIDDEN_SIZE=4096 NUM_LAYERS=36`   |
| 18.4B      | `NUM_ATTENTION_HEADS=48 HIDDEN_SIZE=6144 NUM_LAYERS=40`   |
| 39.1B      | `NUM_ATTENTION_HEADS=64 HIDDEN_SIZE=8192 NUM_LAYERS=48`   |
| 76.1B      | `NUM_ATTENTION_HEADS=80 HIDDEN_SIZE=10240 NUM_LAYERS=60`  |
| 145.6B     | `NUM_ATTENTION_HEADS=96 HIDDEN_SIZE=12288 NUM_LAYERS=80`  |
| 310.1B     | `NUM_ATTENTION_HEADS=128 HIDDEN_SIZE=16384 NUM_LAYERS=96` |

## 4. Appendix

### 4.1. Benchmark mode

To run in benchmark mode (i.e., train only, no validation and test), apply these changes to `2.distributed-training.sbatch` when calling `pretrain_gpt.py`:

```diff
-        --eval-iters 40 \
-        --eval-interval 1000 \
-        --split 98,2,0 \
+        --eval-iters 0 \
+        --split 100,0,0 \
```

Incorrect settings will cause this error message to appear in the Slurm output:

```text
Traceback (most recent call last):
  File "/workspace/Megatron-LM/pretrain_gpt.py", line 198, in <module>
    pretrain(train_valid_test_datasets_provider,
  File "/workspace/Megatron-LM/megatron/training.py", line 227, in pretrain
    = build_train_valid_test_data_iterators(
  File "/workspace/Megatron-LM/megatron/training.py", line 1283, in build_train_valid_test_data_iterators
    build_train_valid_test_data_loaders(
  File "/workspace/Megatron-LM/megatron/training.py", line 1244, in build_train_valid_test_data_loaders
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
  File "/workspace/Megatron-LM/megatron/training.py", line 1214, in build_train_valid_test_datasets
    return build_train_valid_test_datasets_provider(train_val_test_num_samples)
  File "/workspace/Megatron-LM/pretrain_gpt.py", line 186, in train_valid_test_datasets_provider
    ).build()
  File "/workspace/Megatron-LM/megatron/core/datasets/blended_megatron_dataset_builder.py", line 56, in build
    return self._build_blended_dataset_splits()
  File "/workspace/Megatron-LM/megatron/core/datasets/blended_megatron_dataset_builder.py", line 76, in _build_blended_dataset_splits
    return self._build_megatron_dataset_splits(blend[0], split, self.sizes)
  File "/workspace/Megatron-LM/megatron/core/datasets/blended_megatron_dataset_builder.py", line 216, in _build_megatron_dataset_splits
    self.build_generic_dataset(
  File "/workspace/Megatron-LM/megatron/core/datasets/blended_megatron_dataset_builder.py", line 258, in build_generic_dataset
    dataset = cls(*args)
  File "/workspace/Megatron-LM/megatron/core/datasets/gpt_dataset.py", line 68, in __init__
    super().__init__(indexed_dataset, indexed_indices, num_samples, index_split, config)
  File "/workspace/Megatron-LM/megatron/core/datasets/megatron_dataset.py", line 42, in __init__
    assert num_samples > 0
AssertionError
```

### 4.2. Adjust training steps

By default, the .sbatch scripts specify the number of samples, then the number of training steps equals to `--train_samples` / `--global-batch-size`. To directly specify the number of steps, apply these changes to `2.distributed-training.sbatch` when calling `pretrain_gpt.py`. Note that `samples` and `iters` are mutually exclusive.

```diff
-        --train-samples 146484375 \
-        --lr-decay-samples 126953125 \
-        --lr-warmup-samples 183105 \
+        --train-iters 50 \
+        --lr-decay-iters 45 \
+        --lr-warmup-iters 2 \
```
=======

Following the same pattern, you can train other models. Pretraining scripts for models like 
Bert, ICT, and T5 are already included in the Megatron-LM container under `/workspace/Megatron-LM`. 
