# Retained legacy paths

Reference architectures have moved to [`architectures/`](../architectures/). The following assets remain at their original paths for compatibility with the SageMaker HyperPod console and existing documentation links:

- [`5.sagemaker-hyperpod/LifecycleScripts`](./5.sagemaker-hyperpod/LifecycleScripts) — used by [`architectures/sagemaker-hyperpod-slurm`](../architectures/sagemaker-hyperpod-slurm)
- [`7.sagemaker-hyperpod-eks/LifecycleScripts`](./7.sagemaker-hyperpod-eks/LifecycleScripts) — used by [`architectures/sagemaker-hyperpod-eks`](../architectures/sagemaker-hyperpod-eks)
- [`5.sagemaker-hyperpod/Extensions`](./5.sagemaker-hyperpod/Extensions): extension scripts for HyperPod Slurm clusters.
- [`5.sagemaker-hyperpod/patching-backup.sh`](./5.sagemaker-hyperpod/patching-backup.sh): backup and restore script for HyperPod Slurm cluster patching.

Do not add unrelated new content here. These assets can move to their `architectures/` homes once their console and documentation references have been updated.
