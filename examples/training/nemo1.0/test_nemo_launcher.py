import pytest
import os


def test_0_aws_nemo_megatron(docker_build, docker_run):
    img = docker_build('aws-nemo-megatron', '0.NemoMegatron-aws-optimized.Dockerfile')
    docker_run(img, ['python3', '-c', 'import torch'])
