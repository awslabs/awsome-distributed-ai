# Latency and training results from the reference run

Run date: 2026-09-28, on a SageMaker HyperPod EKS cluster in us-east-1.
Serving used an `ml.g6e.8xlarge`, and training used an `ml.g6e.xlarge` (one NVIDIA L40S each).

## Capture overhead (`scripts/pod.sh <endpoint> bench ...`, run from inside the model pod)

These are the same 400 routing requests, sent either straight to vLLM (port 8000) or through the capture proxy (port 8081).

| Path | Concurrency | p50 | p99 | Throughput |
|---|---|---|---|---|
| Direct to vLLM (8000), no capture | 8 | 200.7 ms | 249.4 ms | 39.1 req/s |
| Through capture proxy (8081) | 8 | 200.0 ms | 249.8 ms | 39.1 req/s |
| Direct to vLLM (8000), no capture | 1 | 164.4 ms | 206.9 ms | 6.0 req/s |
| Through capture proxy (8081) | 1 | 164.4 ms | 207.0 ms | 6.0 req/s |

The concurrency-8 rows are the average of three rounds. The p50 varied by about ±1 ms from round to round, which is more than the difference between the two paths.

## Capture storage

The capture prefix held 3,955 records in 9.0 MB, about 2.3 KB per record. The uploader wrote one S3 object per 60-second flush interval.

## Fine-tuning job (`manifests/finetune-job.yaml`)

The job fine-tuned the model with LoRA at rank 16 (about 29.9 million trainable parameters) for 2 epochs over 900 examples.

```
{'loss': 0.2484, 'grad_norm': 3.387, 'learning_rate': 3.91e-05, 'epoch': 0.09}
{'loss': 0.018, 'grad_norm': 0.0037, 'learning_rate': 8.26e-05, 'epoch': 0.18}
...
{'loss': 0.0, 'grad_norm': 0.000105, 'learning_rate': 2.93e-07, 'epoch': 1.95}
{'train_runtime': 59.68, 'train_samples_per_second': 30.16, 'train_loss': 0.0123, 'epoch': 2.0}
[done] merged model uploaded to s3://<bucket>/models/qwen2.5-3b-router-v2/
[HyperPodElasticAgent] ... Transitioned from AgentState.RUNNING to AgentState.COMPLETED
```

- **Training time:** 60 seconds of training. Submission to completion took about 7 minutes, of which 3 min 42 s was the first pull of the 10.9 GB `vllm/vllm-openai:v0.10.0` image.
- **Fine-tuned endpoint:** ready 2 min 19 s after `kubectl apply`, on the node the job had just released. The image was already cached there.
