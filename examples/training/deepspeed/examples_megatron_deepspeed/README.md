# Megatron-DeepSpeed Test Cases <!-- omit in toc -->
[DeepSpeed version of NVIDIA's Megatron-LM](https://github.com/microsoft/Megatron-DeepSpeed/tree/main) adds additional support for several features such as MoE model training, Curriculum Learning, 3D Parallelism, and others to [DeepSpeed](https://github.com/microsoft/DeepSpeed) framework. The `examples_deepspeed` directory includes example scripts about the features supported by DeepSpeed.

## 1. Preparation

You need to follow steps in `../README.md` to prepare AWS-optimized DeepSpeed container. Also, set the following environment variables to run the test cases:

```bash
export APPS_PATH=/fsx/apps
export ENROOT_IMAGE=$APPS_PATH/deepspeed.sqsh
export FSX_PATH=/fsx
export MODEL_PATH=$FSX_PATH/deepspeed
export TEST_CASE_PATH=${HOME}/18.deepspeed  # where you copy the test case or set to your test case path
cd $TEST_CASE_PATH                          # Note that we assume that you are here during the following command executions
```

Then clone the project repository:

```bash
git clone https://github.com/microsoft/Megatron-DeepSpeed
```

Proceed to each example sub-directory once the set up has completed.