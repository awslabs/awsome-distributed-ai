# Retained legacy paths

Everything that used to live under `1.architectures/` has moved to
[`architectures/`](../architectures/) — except the two `LifecycleScripts`
trees below, which are **deliberately kept at their original paths** because
the SageMaker HyperPod console references them at these exact locations on
`main`:

- [`5.sagemaker-hyperpod/LifecycleScripts`](./5.sagemaker-hyperpod/LifecycleScripts) — used by [`architectures/sagemaker-hyperpod-slurm`](../architectures/sagemaker-hyperpod-slurm)
- [`7.sagemaker-hyperpod-eks/LifecycleScripts`](./7.sagemaker-hyperpod-eks/LifecycleScripts) — used by [`architectures/sagemaker-hyperpod-eks`](../architectures/sagemaker-hyperpod-eks)

Do not add new content here. Once the HyperPod service team repoints the
console to a pinned release (or to the new paths), these trees will move to
their `architectures/` homes as well.
