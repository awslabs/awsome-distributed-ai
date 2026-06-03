#!/usr/bin/env  python3
import pytest
import os

def test_img_megatron_training(docker_build, docker_run):
    print(f"module file {os.path.dirname(__file__)}")
    print(f"cwd {os.getcwd()}")
    img = docker_build('megatron-training', 'megatron-lm.Dockerfile')
    docker_run(img, ['python3', '-c', 'import torch'])
